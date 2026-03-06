"""PatchTST training loop for Kronos (Futures) sleeve.

Targets DEVICE_HEAVY (cuda:0, RTX 3090 Ti).
Strict temporal split: Train 2018-2024, Val 2025.
Uses StrictEmbargoDataset to prevent stride overlap leakage (Blind Spot 3).
pin_memory=True with num_workers=4 (offline training on cuda:0 — safe).

Acceptance Criteria:
  - Directional Accuracy: 52.5% - 54.0%
  - R²: 0.01 - 0.05
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from algaie.models.tsfm.patchtst import ContinuousPatchTST
from algaie.models.utils.dataset import StrictEmbargoDataset

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────
DEFAULT_DEVICE = "cuda:0"
ARTIFACT_DIR = Path("backend/artifacts/models/kronos")


# ═══════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════

def directional_accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Fraction of predictions with correct sign (direction)."""
    correct = ((y_pred > 0) == (y_true > 0)).float()
    return correct.mean().item()


def r_squared(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    """Coefficient of determination (R²)."""
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    if ss_tot < 1e-8:
        return 0.0
    return 1.0 - (ss_res / ss_tot).item()


# ═══════════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════════

def load_futures_data(
    data_dir: Path = Path("backend/data_canonical/ohlcv_adj"),
    symbol: str = "es_fut",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load preprocessed futures data as stationary log-returns.

    Converts raw OHLCV close prices to log-returns for both input
    and target to enforce weak stationarity. The model predicts
    next-step log-return from past log-returns.

    Returns
    -------
    data : torch.Tensor
        Shape ``(T-2, 1)`` — past log-returns (stationary input).
    targets : torch.Tensor
        Shape ``(T-2,)`` — next-step log-returns (stationary target).
    """
    import pandas as pd

    parquet_path = data_dir / f"{symbol}_ohlcv.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Data not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    close = df["close"].values.astype(np.float32)

    # Enforce stationarity: log-returns r_t = ln(P_t / P_{t-1})
    log_returns = np.log(close[1:] / close[:-1]).astype(np.float32)

    # Input: past log-returns, Target: next log-return
    # data[t] = r_t, targets[t] = r_{t+1}
    data = torch.from_numpy(log_returns[:-1]).unsqueeze(-1)  # (T-2, 1)
    targets = torch.from_numpy(log_returns[1:])               # (T-2,)

    logger.info("LOAD  %s — %d rows, log-return range [%.6f, %.6f], std=%.6f",
                parquet_path, len(data),
                log_returns.min(), log_returns.max(), log_returns.std())
    return data, targets


# ═══════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════

# BPS scaling: multiply log-returns by 10,000 before MSE to prevent
# gradient underflow. Daily log-returns are O(1e-4); squaring gives
# O(1e-8) which underflows AdamW in mixed precision.
BPS_SCALE = 10_000

def train_kronos(
    data_dir: Path = Path("backend/data_canonical/ohlcv_adj"),
    symbol: str = "es_fut",
    device_str: str = DEFAULT_DEVICE,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-4,
    weight_decay: float = 1e-5,
    seq_len: int = 512,
    patch_len: int = 16,
    stride: int = 8,
    d_model: int = 128,
    n_heads: int = 8,
    n_layers: int = 3,
    patience: int = 10,
    output_dir: Path = ARTIFACT_DIR,
) -> dict:
    """Full PatchTST training pipeline.

    Parameters
    ----------
    Various hyperparameters — see defaults above.

    Returns
    -------
    dict with training history, final metrics, and model path.
    """
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    logger.info("DEVICE %s", device)

    # ── Load data ────────────────────────────────────────────────────
    data, targets = load_futures_data(data_dir, symbol)

    # ── Create embargoed train/val datasets ──────────────────────────
    train_ds, val_ds = StrictEmbargoDataset.create_train_val_pair(
        data=data.squeeze(-1),  # (T, ) for single channel
        targets=targets,
        train_fraction=0.75,  # ~2018-2023 = train, ~2024-2025 = val
        seq_len=seq_len,
        stride=stride,
        forecast_horizon=1,
    )

    # pin_memory=True is safe for offline training on cuda:0 (Blind Spot 1)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    logger.info(
        "DATA  train=%d samples, val=%d samples, embargo=%d rows dropped",
        len(train_ds), len(val_ds), train_ds.embargo_rows_dropped,
    )

    # ── Build model ──────────────────────────────────────────────────
    model = ContinuousPatchTST(
        c_in=1,
        seq_len=seq_len,
        patch_len=patch_len,
        stride=stride,
        forecast_horizon=1,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    logger.info("MODEL ContinuousPatchTST — %d parameters", param_count)

    # ── Optimizer + Scheduler ────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # ── Training loop ────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "val_da": [], "val_r2": []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        # ── Train ────────────────────────────────────────────────────
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)

            pred = model(x_batch)
            loss = ContinuousPatchTST.mse_loss(
                pred.squeeze(-1) * BPS_SCALE, y_batch * BPS_SCALE
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_losses.append(loss.item())

        scheduler.step()
        avg_train_loss = np.mean(train_losses) if train_losses else 0.0

        # ── Validate ─────────────────────────────────────────────────
        model.eval()
        val_preds, val_trues = [], []
        val_losses = []

        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)

                pred = model(x_batch)
                loss = ContinuousPatchTST.mse_loss(
                    pred.squeeze(-1) * BPS_SCALE, y_batch * BPS_SCALE
                )
                val_losses.append(loss.item())

                val_preds.append(pred.squeeze(-1).cpu())
                val_trues.append(y_batch.cpu())

        avg_val_loss = np.mean(val_losses) if val_losses else 0.0

        if val_preds:
            all_preds = torch.cat(val_preds)
            all_trues = torch.cat(val_trues)
            da = directional_accuracy(all_preds, all_trues)
            r2 = r_squared(all_preds, all_trues)
        else:
            da, r2 = 0.0, 0.0

        elapsed = time.time() - t0
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_da"].append(da)
        history["val_r2"].append(r2)

        logger.info(
            "EPOCH %3d/%d  train_loss=%.6f  val_loss=%.6f  DA=%.3f  R²=%.4f  "
            "lr=%.2e  [%.1fs]",
            epoch, epochs, avg_train_loss, avg_val_loss, da, r2,
            scheduler.get_last_lr()[0], elapsed,
        )

        # ── Early stopping ───────────────────────────────────────────
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0

            output_dir.mkdir(parents=True, exist_ok=True)
            model_path = output_dir / "patchtst_kronos_best.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_loss": best_val_loss,
                "da": da,
                "r2": r2,
                "config": {
                    "seq_len": seq_len, "patch_len": patch_len, "stride": stride,
                    "d_model": d_model, "n_heads": n_heads, "n_layers": n_layers,
                    "residual_std": float(torch.cat(val_trues).std().item()) if val_trues else 0.014,
                },
            }, model_path)
            logger.info("SAVE  best model → %s", model_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("STOP  early stopping at epoch %d (patience=%d)", epoch, patience)
                break

    # ── Acceptance criteria check ────────────────────────────────────
    final_da = history["val_da"][-1] if history["val_da"] else 0.0
    final_r2 = history["val_r2"][-1] if history["val_r2"] else 0.0

    da_pass = 0.525 <= final_da <= 0.54
    r2_pass = 0.01 <= final_r2 <= 0.05

    if final_da > 0.55:
        logger.warning(
            "⚠ DA=%.3f exceeds 55%% — POSSIBLE DATA LEAKAGE. "
            "Review embargo and temporal split integrity.", final_da
        )

    logger.info("=" * 60)
    logger.info("ACCEPTANCE  DA=%.3f [%s]  R²=%.4f [%s]",
                final_da, "PASS" if da_pass else "REVIEW",
                final_r2, "PASS" if r2_pass else "REVIEW")
    logger.info("=" * 60)

    return {
        "model_path": str(output_dir / "patchtst_kronos_best.pt"),
        "history": history,
        "final_da": final_da,
        "final_r2": final_r2,
        "da_pass": da_pass,
        "r2_pass": r2_pass,
        "epochs_trained": len(history["train_loss"]),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = train_kronos()
    print(f"\nFinal DA: {result['final_da']:.3f}")
    print(f"Final R²: {result['final_r2']:.4f}")

    # VRAM cleanup — prevent caching allocator fragmentation when
    # training scripts are chained sequentially (Blind Spot 1).
    import gc
    del result
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
