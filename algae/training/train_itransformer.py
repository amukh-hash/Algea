"""
Train iTransformer — Multivariate StatArb model.

Uses Huber Loss + TimeSeriesSplit (no data leakage) on 5-min sector ETF
log-returns.  Targets: forward H-step mean-reversion magnitude.

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
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# ── Defaults ─────────────────────────────────────────────────────────────
DEFAULTS = {
    "num_variates": 6,
    "lookback_len": 60,
    "pred_len": 12,  # 12 × 5min = 1 hour horizon
    "d_model": 256,
    "n_heads": 8,
    "e_layers": 3,
    "dropout": 0.1,
    "epochs": 50,
    "batch_size": 64,
    "lr": 3e-4,
    "weight_decay": 1e-5,
    "patience": 10,
}


def build_targets(
    features: np.ndarray, pred_len: int = 12
) -> np.ndarray:
    """Compute forward H-step mean-reversion targets.

    For each sliding window ending at time t, compute the mean log-return
    over [t+1 … t+pred_len] per variate — the mean-reversion magnitude.

    Parameters
    ----------
    features : np.ndarray
        Shape ``[num_samples, lookback_len, num_variates]``.
    pred_len : int
        Forward horizon.

    Returns
    -------
    np.ndarray
        Shape ``[num_usable_samples, num_variates, pred_len]``.
    """
    num_samples, lookback_len, num_variates = features.shape
    usable = num_samples - pred_len

    if usable <= 0:
        raise ValueError(
            f"Not enough samples ({num_samples}) for pred_len={pred_len}"
        )

    targets = np.zeros((usable, num_variates, pred_len), dtype=np.float32)
    for i in range(usable):
        # Forward window: the log-returns from t+1 to t+pred_len
        # Each feature window ends at the last row of that window
        # The "next" features start at features[i+1]
        for h in range(pred_len):
            if i + 1 + h < num_samples:
                # Take the last timestep of the next window as the target
                targets[i, :, h] = features[i + 1 + h, -1, :]

    return targets


def train_itransformer(
    features_path: str,
    output_path: str = "backend/artifacts/model_weights/itransformer_v1.pt",
    device: str = "cuda:0",
    **kwargs,
) -> dict:
    """Train the iTransformer on StatArb features.

    Parameters
    ----------
    features_path : str
        Path to ``.npy`` file from ``statarb_pipeline.py``.
    output_path : str
        Where to save the checkpoint.
    device : str
        Training device (should be cuda:0 for 3090 Ti).

    Returns
    -------
    dict
        Training summary with final loss, best loss, epochs.
    """
    from algae.models.tsfm.itransformer import iTransformer

    cfg = {**DEFAULTS, **kwargs}
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    logger.info("Training iTransformer on %s", dev)

    # ── Load data ────────────────────────────────────────────────────
    features = np.load(features_path)
    logger.info("Loaded features: %s", features.shape)

    num_variates = features.shape[2]
    lookback_len = features.shape[1]
    pred_len = cfg["pred_len"]

    # Build targets
    targets = build_targets(features, pred_len)
    features = features[: len(targets)]  # Trim to match

    # ── TimeSeriesSplit (no data leakage) ────────────────────────────
    n = len(features)
    split_idx = int(n * 0.8)
    train_X = torch.tensor(features[:split_idx], dtype=torch.float32)
    train_Y = torch.tensor(targets[:split_idx], dtype=torch.float32)
    val_X = torch.tensor(features[split_idx:], dtype=torch.float32)
    val_Y = torch.tensor(targets[split_idx:], dtype=torch.float32)

    train_ds = TensorDataset(train_X, train_Y)
    val_ds = TensorDataset(val_X, val_Y)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False)

    logger.info("Train: %d samples, Val: %d samples", len(train_ds), len(val_ds))

    # ── Model ────────────────────────────────────────────────────────
    model = iTransformer(
        num_variates=num_variates,
        lookback_len=lookback_len,
        pred_len=pred_len,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        e_layers=cfg["e_layers"],
        dropout=cfg["dropout"],
    ).to(dev)

    # ── Huber Loss (Smooth L1) — handles fat-tailed outliers ─────────
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    # ── Training loop ────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg["epochs"]):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch in train_loader:
            X_batch = X_batch.to(dev)
            Y_batch = Y_batch.to(dev)

            optimizer.zero_grad()
            pred = model(X_batch)  # [B, N, pred_len]
            loss = criterion(pred, Y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= max(len(train_loader), 1)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.inference_mode():
            for X_batch, Y_batch in val_loader:
                X_batch = X_batch.to(dev)
                Y_batch = Y_batch.to(dev)
                pred = model(X_batch)
                val_loss += criterion(pred, Y_batch).item()
        val_loss /= max(len(val_loader), 1)

        scheduler.step()

        logger.info(
            "Epoch %d/%d — train_loss=%.6f  val_loss=%.6f  lr=%.2e",
            epoch + 1, cfg["epochs"], train_loss, val_loss,
            optimizer.param_groups[0]["lr"],
        )

        # Checkpoint best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), out_path)
            logger.info("✓ Saved best checkpoint → %s", out_path)
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

    # Cleanup CUDA
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "status": "ok",
        "checkpoint": str(out_path),
        "best_val_loss": best_val_loss,
        "epochs_completed": epoch + 1,
        "num_variates": num_variates,
        "lookback_len": lookback_len,
        "pred_len": pred_len,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="data_lake/statarb/itransformer_features.npy")
    parser.add_argument("--output", default="backend/artifacts/model_weights/itransformer_v1.pt")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    result = train_itransformer(
        features_path=args.features,
        output_path=args.output,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
    print(result)
