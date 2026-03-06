"""
Offline Training Pipeline: SMoE RankTransformer Ensemble (Sequence 2)
Target: cuda:0 (RTX 3090 Ti) strictly.

Trains 4 expert RankTransformers on the partitioned cross-sectional
equity datasets using Vectorized Pairwise Margin Ranking Loss.

Expert 0: Tech/Growth (XLK+XLC+XLY)
Expert 1: Value/Defensive (XLF+XLE+XLV+XLP+XLU)
Expert 2: Crisis (VIX > 20)
Expert 3: Calm (VIX ≤ 20)
"""
import gc
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SMoE_Trainer")

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# ═══════════════════════════════════════════════════════════════════════════
# 1. Dataset
# ═══════════════════════════════════════════════════════════════════════════

class SMoEDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, mask: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        # Dataset mask: True = padded (ghost stock)
        # RankTransformer mask: 1 = valid, 0 = padding
        # We store as dataset convention and flip in training loop
        self.mask = torch.tensor(mask, dtype=torch.bool)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.mask[idx]


# ═══════════════════════════════════════════════════════════════════════════
# 2. Vectorized Pairwise Margin Ranking Loss
# ═══════════════════════════════════════════════════════════════════════════

class PairwiseRankingLoss(nn.Module):
    """
    Constructs a NxN difference matrix on GPU. For every pair (i,j)
    where stock i outperformed stock j, penalizes if pred[j] > pred[i].
    Scaled by absolute rank difference (top/bottom decile focus).
    """
    def __init__(self, margin: float = 0.05):
        super().__init__()
        self.margin = margin

    def forward(
        self, preds: torch.Tensor, targets: torch.Tensor, pad_mask: torch.Tensor,
    ) -> torch.Tensor:
        # preds: [B, N], targets: [B, N], pad_mask: [B, N] (True = padded)

        # Pairwise difference matrices [B, N, N]
        diff_p = preds.unsqueeze(2) - preds.unsqueeze(1)
        diff_y = targets.unsqueeze(2) - targets.unsqueeze(1)

        # Only penalize pairs where stock i truly beat stock j
        target_greater = (diff_y > 0).float()

        # Hinge: penalize if pred[i] isn't beating pred[j] by margin
        loss = torch.relu(self.margin - diff_p)

        # Scale by absolute rank difference (top/bottom decile focus)
        rank_diff = torch.abs(diff_y)

        # Valid pair mask (both stocks must be real, not padding)
        valid = (~pad_mask).float()  # [B, N]
        valid_pairs = valid.unsqueeze(2) * valid.unsqueeze(1)  # [B, N, N]

        loss = loss * target_greater * valid_pairs * rank_diff

        num_valid = (target_greater * valid_pairs).sum()
        if num_valid > 0:
            return loss.sum() / num_valid
        return torch.tensor(0.0, device=preds.device, requires_grad=True)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Evaluation: Spearman Rank IC
# ═══════════════════════════════════════════════════════════════════════════

def compute_batch_ic(
    preds: np.ndarray, targets: np.ndarray, mask: np.ndarray,
) -> float:
    """Mean Spearman rank correlation across the batch (per day)."""
    ics = []
    for b in range(preds.shape[0]):
        valid_idx = ~mask[b]
        if valid_idx.sum() > 2:
            p = preds[b][valid_idx]
            t = targets[b][valid_idx]
            if np.std(p) < 1e-6 or np.std(t) < 1e-6:
                continue
            corr, _ = spearmanr(p, t)
            if not np.isnan(corr):
                ics.append(corr)
    return float(np.mean(ics)) if ics else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# 4. Training Engine
# ═══════════════════════════════════════════════════════════════════════════

def train_expert(
    expert_id: int,
    data_dir: Path,
    out_dir: Path,
    epochs: int = 35,
):
    """Train a single RankTransformer expert on its partition."""
    from algae.models.ranker.rank_transformer import RankTransformer

    logger.info("=" * 60)
    logger.info("Training Expert %d", expert_id)
    logger.info("=" * 60)

    # Load data
    X_raw = np.load(data_dir / f"X_expert_{expert_id}.npy")
    y_raw = np.load(data_dir / f"y_expert_{expert_id}.npy")
    mask_raw = np.load(data_dir / f"mask_expert_{expert_id}.npy")
    logger.info("  Data: X=%s y=%s mask=%s", X_raw.shape, y_raw.shape, mask_raw.shape)

    # Chronological split (80/20) with 3-day embargo (forward return horizon)
    split_idx = int(len(y_raw) * 0.8)
    embargo_idx = split_idx + 3

    X_train, y_train, mask_train = X_raw[:split_idx], y_raw[:split_idx], mask_raw[:split_idx]
    X_val, y_val, mask_val = X_raw[embargo_idx:], y_raw[embargo_idx:], mask_raw[embargo_idx:]
    logger.info("  Split: %d train, %d val (3-day embargo)", len(X_train), len(X_val))

    train_loader = DataLoader(
        SMoEDataset(X_train, y_train, mask_train),
        batch_size=16, shuffle=True, drop_last=True,
    )
    val_loader = DataLoader(
        SMoEDataset(X_val, y_val, mask_val),
        batch_size=32, shuffle=False,
    )

    # Initialize model — d_input=8, max_len=512 for our data
    model = RankTransformer(
        d_input=8,
        d_model=128,
        n_head=4,
        n_layers=2,
        dropout=0.15,
        max_len=512,
        pooling="none",
    ).to(DEVICE, dtype=torch.bfloat16)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info("  Model: %s params", f"{total_params:,}")

    criterion = PairwiseRankingLoss(margin=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_ic = -1.0
    best_state = None

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0

        for X_b, y_b, pad_mask_b in tqdm(
            train_loader,
            desc=f"Exp {expert_id} Ep {epoch + 1:02d}/{epochs}",
            leave=False,
        ):
            X_b = X_b.to(DEVICE, dtype=torch.bfloat16)
            y_b = y_b.to(DEVICE, dtype=torch.float32)
            pad_mask_b = pad_mask_b.to(DEVICE)

            optimizer.zero_grad()

            # Zero out padded features
            X_b[pad_mask_b] = 0.0

            # RankTransformer expects mask: 1=valid, 0=padding
            valid_mask = (~pad_mask_b).float()

            outputs = model(X_b, mask=valid_mask)

            # Extract score [B, N, 1] → [B, N]
            scores = outputs["score"].squeeze(-1).float()

            loss = criterion(scores, y_b, pad_mask_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()

        # --- Validate ---
        model.eval()
        val_ics = []

        with torch.inference_mode():
            for X_b, y_b, pad_mask_b in val_loader:
                X_b = X_b.to(DEVICE, dtype=torch.bfloat16)
                y_b = y_b.to(DEVICE, dtype=torch.float32)
                pad_mask_b = pad_mask_b.to(DEVICE)

                X_b[pad_mask_b] = 0.0
                valid_mask = (~pad_mask_b).float()

                outputs = model(X_b, mask=valid_mask)
                scores = outputs["score"].squeeze(-1).float()

                ic = compute_batch_ic(
                    scores.cpu().numpy(),
                    y_b.cpu().numpy(),
                    pad_mask_b.cpu().numpy(),
                )
                val_ics.append(ic)

        avg_train_loss = train_loss / max(len(train_loader), 1)
        avg_val_ic = float(np.mean(val_ics)) if val_ics else 0.0

        logger.info(
            "Epoch %02d | Train Loss: %.4f | Val Spearman IC: %.4f",
            epoch + 1, avg_train_loss, avg_val_ic,
        )

        if avg_val_ic > best_ic:
            best_ic = avg_val_ic
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            logger.info("  >>> NEW BEST Expert %d (IC: %.4f)", expert_id, best_ic)

    # Save best checkpoint
    if best_state is not None:
        save_path = out_dir / f"expert_{expert_id}.pt"
        torch.save(best_state, save_path)
        logger.info(
            "Expert %d COMPLETE | Best IC: %.4f | Saved: %s (%.1f KB)",
            expert_id, best_ic, save_path.name,
            save_path.stat().st_size / 1024,
        )
    else:
        logger.warning("Expert %d: No valid state to save!", expert_id)

    del model, optimizer, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()

    return best_ic


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    data_dir = Path("data_lake/smoe_training")
    out_dir = Path("backend/artifacts/model_weights")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for i in range(4):
        if (data_dir / f"X_expert_{i}.npy").exists():
            ic = train_expert(expert_id=i, data_dir=data_dir, out_dir=out_dir, epochs=35)
            results[i] = ic
        else:
            logger.error("Data for Expert %d missing!", i)

    logger.info("=" * 60)
    logger.info("SEQUENCE 2 TRAINING COMPLETE")
    for eid, ic in results.items():
        logger.info("  Expert %d: Best IC = %.4f", eid, ic)
    logger.info("=" * 60)
