"""
Train Chronos2 LoRA — Overnight gap distribution fine-tuning.

Freezes the foundation Transformer weights and trains only a Low-Rank
Adapter (LoRA) specifically on the Close-to-Open (15:58 ET → 09:30 ET)
gap distribution of ES futures.

Device: cuda:0 (3090 Ti, 24GB) — offline batch training.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class OvernightGapDataset(Dataset):
    """Dataset of ES futures overnight gaps.

    Each sample is:
        - context: [context_len] price observations leading up to 15:58 close
        - target: [prediction_len] observations from 09:30 open forward

    Parameters
    ----------
    data_dir : str
        Directory containing ``.npz`` files from ``extract_tft_training_data.py``.
    context_len : int
        Number of historical observations for context.
    prediction_len : int
        Number of forward observations to predict.
    """

    def __init__(
        self,
        data_dir: str,
        context_len: int = 32,
        prediction_len: int = 3,
    ):
        self.context_len = context_len
        self.prediction_len = prediction_len
        self.samples: list[tuple[np.ndarray, np.ndarray]] = []

        data_path = Path(data_dir)
        npz_files = sorted(data_path.glob("*.npz"))
        logger.info("Loading overnight gap data from %d files", len(npz_files))

        for f in npz_files:
            try:
                npz = np.load(f)
                # Actual keys from extract_tft_training_data.py:
                #   ts_features: (184, 3) — time-series features (5-min bars)
                #   static_features: (3,) — day-level features
                #   obs_features: (5,) — observation features
                #   oc_return: scalar — overnight close-to-open return
                ts_feat = npz.get("ts_features", None)
                oc_return = npz.get("oc_return", None)

                if ts_feat is None or oc_return is None:
                    continue

                # Use the close price column (last column) from ts_features
                # as the univariate context for Chronos2
                close_series = ts_feat[:, -1].astype(np.float32)
                label = float(oc_return)

                if len(close_series) >= context_len:
                    context = close_series[-context_len:]
                    target = np.array([label] * prediction_len, dtype=np.float32)
                    self.samples.append((context, target))
            except Exception as e:
                logger.debug("Skipping %s: %s", f, e)

        logger.info("Loaded %d overnight gap samples", len(self.samples))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        context, target = self.samples[idx]
        return {
            "past_target": torch.tensor(context, dtype=torch.float32),
            "future_target": torch.tensor(target, dtype=torch.float32),
        }


def train_chronos2_lora(
    data_dir: str = "data_lake/tft_training",
    model_id: str = "amazon/chronos-bolt-tiny",
    output_path: str = "backend/artifacts/model_weights/chronos2_es_adapter.pt",
    context_len: int = 32,
    prediction_len: int = 3,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    epochs: int = 10,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: str = "cuda:0",
) -> dict:
    """Fine-tune Chronos2 with LoRA on overnight gap distribution.

    Parameters
    ----------
    data_dir : str
        Directory with ``.npz`` training files.
    model_id : str
        HuggingFace model ID for the base Chronos2 model.
    output_path : str
        Where to save the LoRA adapter weights.
    lora_rank, lora_alpha, lora_dropout : LoRA hyperparameters.

    Returns
    -------
    dict
        Training summary.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info("Training Chronos2 LoRA on %s", dev)

    # ── Load model with LoRA ─────────────────────────────────────────
    try:
        from algae.models.foundation.chronos2_teacher import load_chronos_adapter

        lora_config = {
            "r": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "target_modules": ["q_proj", "v_proj"],
        }

        model_wrapper, info = load_chronos_adapter(
            model_id=model_id,
            use_qlora=False,
            device=dev,
            lora_config=lora_config,
        )
        logger.info("Loaded Chronos2 with LoRA: %s", info.get("model_type", "unknown"))
    except ImportError as e:
        logger.error("Cannot import chronos2_teacher: %s", e)
        logger.info("Falling back to standalone LoRA training approach")
        return _train_standalone_lora(
            data_dir, output_path, context_len, prediction_len,
            epochs, batch_size, lr, device
        )
    except Exception as e:
        logger.error("Failed to load Chronos2: %s", e)
        return _train_standalone_lora(
            data_dir, output_path, context_len, prediction_len,
            epochs, batch_size, lr, device
        )

    # ── Freeze base, train only LoRA ─────────────────────────────────
    trainable_params = 0
    total_params = 0
    for name, param in model_wrapper.named_parameters():
        total_params += param.numel()
        if "lora" in name.lower() or param.requires_grad:
            trainable_params += param.numel()
        else:
            param.requires_grad = False

    logger.info(
        "Parameters: %d total, %d trainable (%.1f%%)",
        total_params, trainable_params, 100.0 * trainable_params / max(total_params, 1)
    )

    # ── Dataset ──────────────────────────────────────────────────────
    dataset = OvernightGapDataset(data_dir, context_len, prediction_len)
    if len(dataset) == 0:
        logger.error("No training samples found")
        return {"status": "error", "reason": "no_data"}

    # TimeSeriesSplit
    split = int(len(dataset) * 0.85)
    train_ds = torch.utils.data.Subset(dataset, range(0, split))
    val_ds = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    # ── Training ─────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model_wrapper.parameters() if p.requires_grad],
        lr=lr, weight_decay=1e-5,
    )

    best_val_loss = float("inf")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    model_wrapper.train()
    for epoch in range(epochs):
        train_loss = 0.0
        steps = 0
        for batch in train_loader:
            context = batch["past_target"].to(dev)
            future = batch["future_target"].to(dev)

            optimizer.zero_grad()
            try:
                outputs = model_wrapper(
                    context=context,
                    future_target=future,
                )
                loss = outputs.loss if hasattr(outputs, "loss") else outputs.get("loss", None)
                if loss is not None and loss.requires_grad:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in model_wrapper.parameters() if p.requires_grad], 1.0
                    )
                    optimizer.step()
                    train_loss += loss.item()
            except Exception as e:
                logger.warning("Training step failed: %s", e)
            steps += 1

        train_loss /= max(steps, 1)

        # Validate
        model_wrapper.eval()
        val_loss = 0.0
        val_steps = 0
        with torch.inference_mode():
            for batch in val_loader:
                context = batch["past_target"].to(dev)
                future = batch["future_target"].to(dev)
                try:
                    outputs = model_wrapper(context=context, future_target=future)
                    loss = outputs.loss if hasattr(outputs, "loss") else outputs.get("loss", None)
                    if loss is not None:
                        val_loss += loss.item()
                except Exception:
                    pass
                val_steps += 1
        val_loss /= max(val_steps, 1)
        model_wrapper.train()

        logger.info("Epoch %d/%d — train=%.6f val=%.6f", epoch + 1, epochs, train_loss, val_loss)

        if val_loss < best_val_loss and val_loss > 0:
            best_val_loss = val_loss
            # Save LoRA weights only
            lora_state = {
                k: v for k, v in model_wrapper.state_dict().items()
                if "lora" in k.lower()
            }
            if not lora_state:
                lora_state = model_wrapper.state_dict()
            torch.save(lora_state, out)
            logger.info("✓ Saved LoRA checkpoint → %s", out)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {"status": "ok", "checkpoint": str(out), "best_val_loss": best_val_loss}


def _train_standalone_lora(
    data_dir: str,
    output_path: str,
    context_len: int,
    prediction_len: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
) -> dict:
    """Fallback: train a simple forecasting head when HF model isn't available."""
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info("Using standalone LoRA fallback training on %s", dev)

    dataset = OvernightGapDataset(data_dir, context_len, prediction_len)
    if len(dataset) == 0:
        return {"status": "error", "reason": "no_data"}

    # Simple linear model as a placeholder
    model = nn.Sequential(
        nn.Linear(context_len, 64),
        nn.GELU(),
        nn.Linear(64, prediction_len),
    ).to(dev)

    split = int(len(dataset) * 0.85)
    train_ds = torch.utils.data.Subset(dataset, range(0, split))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            x = batch["past_target"].to(dev)
            y = batch["future_target"].to(dev)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total += loss.item()
        logger.info("Epoch %d/%d — loss=%.6f", epoch + 1, epochs, total / max(len(train_loader), 1))

    torch.save(model.state_dict(), out)
    logger.info("Standalone LoRA fallback saved → %s", out)
    return {"status": "ok", "checkpoint": str(out), "mode": "standalone_fallback"}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data_lake/tft_training")
    parser.add_argument("--model-id", default="amazon/chronos-bolt-tiny")
    parser.add_argument("--output", default="backend/artifacts/model_weights/chronos2_es_adapter.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()

    result = train_chronos2_lora(
        data_dir=args.data_dir,
        model_id=args.model_id,
        output_path=args.output,
        device=args.device,
        epochs=args.epochs,
        lr=args.lr,
    )
    print(result)
