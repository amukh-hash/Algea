"""Offline training script for the TFT Core Reversal model.

Weekend Training Protocol — execute on RTX 3090 Ti (device 0).

**Critical Invariants:**
1. Feature window is strictly 18:00 → 09:20 EST (184 five-minute bars).
   If a single 09:25 or 09:30 bar leaks into X, the model will overfit
   to the RTH open and hemorrhage capital in live trading.
2. Labels are the Open-to-Close return: (Close_16:00 - Open_09:30) / Open_09:30.
3. Loss function is Quantile (Pinball) Loss for [P10, P50, P90].
   MSE/MAE will collapse the uncertainty spread and fail the killswitch.
4. Save raw state_dict, NOT the compiled graph. torch.compile happens
   at runtime on the 4070 Super via GPUProcessSupervisor.

Usage
-----
    python scripts/train_tft_gap.py \\
        --data-dir data_lake/es_futures/ \\
        --epochs 100 \\
        --lr 1e-4 \\
        --device cuda:0 \\
        --output artifacts/models/tft_core_prod.pt
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.ml_platform.models.tft_gap.model import TemporalFusionTransformer

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)


# ═══════════════════════════════════════════════════════════════════════
# Quantile (Pinball) Loss — DO NOT REPLACE WITH MSE/MAE
# ═══════════════════════════════════════════════════════════════════════

QUANTILES = [0.10, 0.50, 0.90]


def quantile_loss(
    preds: torch.Tensor,
    target: torch.Tensor,
    quantiles: list[float] = QUANTILES,
) -> torch.Tensor:
    """Pinball loss for quantile regression.

    Parameters
    ----------
    preds : Tensor [B, 3]
        Model output: [P10, P50, P90] quantile predictions.
    target : Tensor [B, 1]
        Ground truth Open-to-Close return.
    quantiles : list[float]
        Quantile levels corresponding to each head.

    Returns
    -------
    Scalar loss tensor.

    Notes
    -----
    For quantile q and error e = target - pred:
      loss = max(q * e, (q - 1) * e)

    This asymmetric penalty ensures each head learns its designated
    quantile rather than collapsing to the mean.
    """
    losses = []
    for i, q in enumerate(quantiles):
        error = target - preds[:, i].unsqueeze(1)
        loss = torch.max(q * error, (q - 1) * error)
        losses.append(loss)
    return torch.cat(losses, dim=1).mean()


# ═══════════════════════════════════════════════════════════════════════
# Dataset — 09:20 Hard Cutoff Enforcement
# ═══════════════════════════════════════════════════════════════════════

class GapReversalDataset(Dataset):
    """Historical /ES gap reversal dataset.

    Each sample contains:
      X_ts:     [184, 3] — overnight 5-min bars (log_return, vol_norm, vwap_norm)
      X_static: [3]      — day_of_week, is_opex, macro_event_id
      X_obs:    [5]      — gap_proxy_pct, nikkei, eurostoxx, zn_drift, vix
      Y:        [1]      — Open-to-Close return (label)

    Parameters
    ----------
    data_dir : Path
        Directory containing pre-processed .npz files from the data pipeline.
        Each .npz must contain: ts_features, static_features, obs_features, oc_return.
    """

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)
        self.samples: list[dict] = []
        self._load_data()

    def _load_data(self) -> None:
        """Load and validate all .npz files in data_dir."""
        npz_files = sorted(self.data_dir.glob("*.npz"))
        if not npz_files:
            raise FileNotFoundError(
                f"No .npz files found in {self.data_dir}. "
                "Run the data extraction pipeline first."
            )

        for f in npz_files:
            data = np.load(f, allow_pickle=True)

            ts = data["ts_features"]  # Expected: [184, 3]
            if ts.shape != (184, 3):
                logger.warning(
                    "SKIPPING %s — ts_features shape %s != (184, 3). "
                    "Possible lookahead contamination.",
                    f.name, ts.shape
                )
                continue

            self.samples.append({
                "ts": ts.astype(np.float32),
                "static": data["static_features"].astype(np.float32),
                "obs": data["obs_features"].astype(np.float32),
                "target": np.array([data["oc_return"]], dtype=np.float32),
            })

        logger.info(
            "Loaded %d valid samples from %s (%d files scanned)",
            len(self.samples), self.data_dir, len(npz_files),
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        s = self.samples[idx]
        return {
            "ts": torch.from_numpy(s["ts"]),
            "static": torch.from_numpy(s["static"]),
            "obs": torch.from_numpy(s["obs"]),
            "target": torch.from_numpy(s["target"]),
        }


# ═══════════════════════════════════════════════════════════════════════
# Training Loop
# ═══════════════════════════════════════════════════════════════════════

def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    logger.info("Training device: %s", device)

    # ── Data ─────────────────────────────────────────────────────────
    dataset = GapReversalDataset(Path(args.data_dir))
    n_total = len(dataset)
    n_val = max(1, int(n_total * 0.15))
    n_train = n_total - n_val

    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=False,  # F5: prevent CUDA fork deadlocks
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=False,
    )

    logger.info("Train: %d samples, Val: %d samples", n_train, n_val)

    # ── Model ────────────────────────────────────────────────────────
    model = TemporalFusionTransformer(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    logger.info(
        "Model params: %d",
        sum(p.numel() for p in model.parameters()),
    )

    # ── Training ─────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            ts = batch["ts"].to(device)
            static = batch["static"].to(device)
            obs = batch["obs"].to(device)
            target = batch["target"].to(device)

            optimizer.zero_grad()
            preds = model(ts, static, obs)
            loss = quantile_loss(preds, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        scheduler.step()
        avg_train = train_loss / max(len(train_loader), 1)

        # ── Validation ───────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                ts = batch["ts"].to(device)
                static = batch["static"].to(device)
                obs = batch["obs"].to(device)
                target = batch["target"].to(device)

                preds = model(ts, static, obs)
                val_loss += quantile_loss(preds, target).item()

        avg_val = val_loss / max(len(val_loader), 1)

        logger.info(
            "Epoch %3d/%d | Train: %.6f | Val: %.6f | LR: %.2e",
            epoch, args.epochs, avg_train, avg_val,
            scheduler.get_last_lr()[0],
        )

        # ── Early Stopping + Best Model Save ─────────────────────────
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            patience_counter = 0

            # Save raw state_dict — NOT the compiled graph
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            torch.save(model.state_dict(), args.output)
            logger.info(
                "✓ Best model saved (val=%.6f) → %s", avg_val, args.output,
            )
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d)",
                    epoch, args.patience,
                )
                break

    # ── Sanity Check: Verify Uncertainty Spread ──────────────────────
    logger.info("Running post-training uncertainty spread sanity check...")
    model.load_state_dict(torch.load(args.output, map_location=device))
    model.eval()

    spreads = []
    with torch.no_grad():
        for batch in val_loader:
            preds = model(
                batch["ts"].to(device),
                batch["static"].to(device),
                batch["obs"].to(device),
            )
            spread = (preds[:, 2] - preds[:, 0]).cpu().numpy()
            spreads.extend(spread.tolist())

    mean_spread = np.mean(spreads)
    min_spread = np.min(spreads)
    logger.info(
        "Uncertainty spread — mean: %.5f, min: %.5f", mean_spread, min_spread,
    )

    if mean_spread < 0.001:
        logger.critical(
            "⚠ ALERT: Mean uncertainty spread < 0.1%%. "
            "This suggests the P10/P90 heads have collapsed. "
            "Verify you are NOT using MSE/MAE loss."
        )
    else:
        logger.info("✓ Uncertainty spread is healthy — killswitch will function.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train TFT Core Reversal model (Pinball Loss)",
    )
    parser.add_argument(
        "--data-dir", type=str, required=True,
        help="Directory with pre-processed .npz training samples",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--output", type=str,
        default="backend/artifacts/models/tft_core_prod.pt",
        help="Path to save the raw state_dict (.pt)",
    )
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
