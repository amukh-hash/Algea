"""
Offline Training Pipeline: Chronos2 LoRA (Sequence 1: Futures Overnight)
Target: cuda:0 (RTX 3090 Ti) strictly. Do not disrupt cuda:1 live inference.
"""
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 1. Hardware Isolation: Protect the live trading GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Chronos2_LoRA_Trainer")

# ═══════════════════════════════════════════════════════════════════════════
# 1. Domain-Specific Dataset
# ═══════════════════════════════════════════════════════════════════════════

class OvernightGapDataset(Dataset):
    """Strictly aligned time-series dataset. No lookahead bias."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        # BPS values are passed directly. The network optimizes well on variance ~64.0
        self.X = torch.tensor(X, dtype=torch.bfloat16)
        self.y = torch.tensor(y, dtype=torch.bfloat16)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Native Chronos-2 expects context as [T] (1D per sample)
        return self.X[idx], self.y[idx]

def get_dataloaders(data_dir: Path, batch_size: int = 32):
    X_raw = np.load(data_dir / "context_windows.npy")
    y_raw = np.load(data_dir / "targets.npy")

    # --- Strict Chronological Split (NEVER SHUFFLE ACROSS BOUNDARY) ---
    total_samples = len(y_raw)
    split_idx = int(total_samples * 0.8)  # Approx 1,000 train, 249 val

    X_train, y_train = X_raw[:split_idx], y_raw[:split_idx]
    X_val, y_val = X_raw[split_idx:], y_raw[split_idx:]

    train_dataset = OvernightGapDataset(X_train, y_train)
    val_dataset = OvernightGapDataset(X_val, y_val)

    # Shuffling Train is mathematically safe because the 32-day windows are self-contained.
    # NEVER shuffle Validation data.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    logger.info(f"Chronological Split: {len(train_dataset)} Train | {len(val_dataset)} Validation")
    return train_loader, val_loader

# ═══════════════════════════════════════════════════════════════════════════
# 2. Pinball (Quantile) Loss Function
# ═══════════════════════════════════════════════════════════════════════════

class PinballLoss(nn.Module):
    """
    Quantile (Pinball) Loss for Probabilistic Forecasting.
    Optimizes the P10, P50 (Median), and P90 bands simultaneously via broadcasting.
    Formula: max(q * err, (q - 1) * err)
    """
    def __init__(self, quantiles: list[float] = [0.10, 0.50, 0.90]):
        super().__init__()
        self.quantiles = torch.tensor(quantiles).view(1, -1)  # Shape: [1, 3]

    def forward(self, preds: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        preds: [Batch, NumQuantiles=3]
        target: [Batch]
        """
        if preds.ndim == 3:
            preds = preds.squeeze(1)  # Flatten if shape is [Batch, 1, 3]

        self.quantiles = self.quantiles.to(preds.device, dtype=preds.dtype)
        target_exp = target.unsqueeze(1)  # [Batch, 1] for broadcasting

        err = target_exp - preds

        # Pinball asymmetrical penalty computed simultaneously
        loss = torch.max(self.quantiles * err, (self.quantiles - 1.0) * err)
        return loss.mean()

# ═══════════════════════════════════════════════════════════════════════════
# 3. Training Loop & Validation Gates
# ═══════════════════════════════════════════════════════════════════════════

def train_chronos2_lora():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Initiating Sequence 1 Training strictly on {device}")

    data_dir = Path("data_lake/chronos2_training")
    train_loader, val_loader = get_dataloaders(data_dir, batch_size=32)

    # --- 1. Model & LoRA Architecture ---
    logger.info("Loading Foundation Model via load_chronos_adapter...")
    from algae.models.foundation.chronos2_teacher import load_chronos_adapter

    # Load native Chronos-2 (120M encoder-only patch-based model)
    # LoRA r=8, alpha=16, targeting q and v attention matrices
    model_wrapper, info = load_chronos_adapter(
        model_id="amazon/chronos-2",
        use_qlora=False,
        device=device,
        lora_config={
            "rank": 8,
            "alpha": 16,
            "dropout": 0.1,
            "target_modules": ["q", "v"],
        },
        eval_mode=False,
    )
    logger.info(f"Model type: {info.get('model_type', 'unknown')}")
    logger.info(f"Wrapper class: {type(model_wrapper).__name__}")

    # Enable the quantile head (disabled by default until validated)
    model_wrapper._enable_q10d_head = True

    # Ensure the custom quantile head remains trainable alongside LoRA
    if hasattr(model_wrapper, "quantile_head"):
        # Cast to bfloat16 to match model dtype (prevents mat1/mat2 dtype mismatch)
        model_wrapper.quantile_head.to(dtype=torch.bfloat16)
        for param in model_wrapper.quantile_head.parameters():
            param.requires_grad = True
        logger.info("Quantile head unfrozen + cast to bfloat16 for joint training")

    model_wrapper.to(device)

    # Print trainable vs total params
    total_params = sum(p.numel() for p in model_wrapper.parameters())
    trainable_params = sum(p.numel() for p in model_wrapper.parameters() if p.requires_grad)
    logger.info(
        f"Trainable Parameters: {trainable_params:,} / {total_params:,} "
        f"({100 * trainable_params / total_params:.2f}%)"
    )

    # SAFETY CHECK: If trainable > 50%, the foundation model wasn't frozen
    if trainable_params / total_params > 0.50:
        logger.critical(
            "HALT: >50%% of parameters are trainable. "
            "Foundation model was NOT frozen. Aborting."
        )
        return

    # --- 2. Optimizer & Scheduler ---
    criterion = PinballLoss(quantiles=[0.10, 0.50, 0.90])
    # 5e-4 is an optimal starting learning rate for LoRA fine-tuning
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model_wrapper.parameters()),
        lr=5e-4, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=40)

    # --- 3. Training Loop ---
    best_coverage_error = float("inf")
    epochs = 40
    output_path = Path("backend/artifacts/model_weights/chronos2_es_adapter.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model_wrapper.train()
        train_loss = 0.0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{epochs}"):
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()

            # Native Chronos-2 training: forward(context, future_target) → NLL loss
            # context: [B, 32], future_target: [B, 1]
            future_target = y_batch.unsqueeze(-1)  # [B] → [B, 1]

            outputs = model_wrapper(
                context=X_batch,
                future_target=future_target,
            )

            # The native model returns NLL loss when future_target is provided
            if hasattr(outputs, "loss") and outputs.loss is not None:
                loss = outputs.loss
            elif isinstance(outputs, dict) and "loss" in outputs:
                loss = outputs["loss"]
            else:
                # Fallback: compute MSE between model output and target
                if hasattr(outputs, "logits"):
                    pred = outputs.logits.mean(dim=-1).squeeze()
                elif isinstance(outputs, dict) and "logits" in outputs:
                    pred = outputs["logits"].mean(dim=-1).squeeze()
                elif isinstance(outputs, torch.Tensor):
                    pred = outputs.squeeze()
                else:
                    logger.warning(f"Unexpected output type: {type(outputs)}")
                    continue
                loss = nn.functional.huber_loss(pred, y_batch)
            loss.backward()

            # Gradient clipping prevents the -396 BPS Evergrande tail from destabilizing weights
            torch.nn.utils.clip_grad_norm_(model_wrapper.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # --- 4. Strict Validation Gate (Coverage Ratio) ---
        model_wrapper.eval()
        val_loss = 0.0
        inside_bands_count = 0
        total_val_samples = 0

        with torch.inference_mode():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)

                # generate() expects [B, T, F] — unsqueeze feature dim
                # Pipeline.predict() internally creates a DataLoader that tries to
                # pin_memory, so context MUST be on CPU (pipeline handles device)
                context_3d = X_val.cpu().unsqueeze(-1).float()  # [B, 32] → [B, 32, 1] on CPU

                # Use generate() for validation quantile estimation
                # Output: [B, 1, 21, 1] — 21 native quantile predictions
                samples = model_wrapper.generate(
                    context=context_3d,
                    prediction_length=1,
                    num_samples=50,  # ignored by deterministic pipeline
                )
                # Reshape: [B, 1, 21, 1] → [B, 21]
                q_all = samples.to(device).squeeze(-1).squeeze(1)  # [B, 21]
                if q_all.ndim == 1:
                    q_all = q_all.unsqueeze(0)

                n_q = q_all.shape[-1]
                if n_q >= 21:
                    # P10=idx2, P50=idx10, P90=idx18 from 21 evenly spaced quantiles
                    q_out = torch.stack([
                        q_all[:, 2],   # ~P10
                        q_all[:, 10],  # ~P50
                        q_all[:, 18],  # ~P90
                    ], dim=-1).float()
                else:
                    q_out = torch.stack([
                        q_all[:, 0],
                        q_all[:, n_q // 2],
                        q_all[:, -1],
                    ], dim=-1).float()

                val_loss += criterion(q_out, y_val.float()).item()

                # Coverage Check: P10 <= Actual <= P90
                within_band = (y_val.float() >= q_out[:, 0]) & (y_val.float() <= q_out[:, 2])
                inside_bands_count += within_band.sum().item()
                total_val_samples += len(y_val)

        avg_train_loss = train_loss / max(len(train_loader), 1)
        avg_val_loss = val_loss / max(len(val_loader), 1)
        coverage_ratio = inside_bands_count / max(total_val_samples, 1)

        logger.info(
            f"Epoch {epoch+1:02d} | "
            f"Train Pinball: {avg_train_loss:.2f} | "
            f"Val Pinball: {avg_val_loss:.2f} | "
            f"Coverage Ratio: {coverage_ratio:.1%} (Target: ~80.0%)"
        )

        # We optimize for a Coverage Ratio exactly at 80% (P90 - P10)
        coverage_error = abs(0.80 - coverage_ratio)
        if coverage_error < best_coverage_error:
            best_coverage_error = coverage_error

            # Save LoRA adapter weights + quantile head
            state_dict = {
                k: v.cpu()
                for k, v in model_wrapper.state_dict().items()
                if "lora_" in k.lower() or "quantile_head" in k.lower()
            }
            torch.save(state_dict, output_path)
            logger.info(f"  >>> New Best Model Saved (Coverage Error: {coverage_error:.3f})")

    logger.info(f"Sequence 1 Training Complete. Best Coverage Error: {best_coverage_error:.3f}")
    logger.info(f"Checkpoint: {output_path}")

if __name__ == "__main__":
    train_chronos2_lora()
