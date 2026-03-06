"""
Daily Vol Surface Grid Accumulator for Sequence 5 Training

Runs daily (via Windows Task Scheduler) to:
1. Fetch live SPY options chain via yfinance
2. Compute IV surface → [5 Tenors × 25 Deltas] grid
3. Append the grid + metadata to data_lake/vrp_history/grids/
4. Maintain a running master tensor for iTransformer training

After ~60-90 trading days, the master tensor will have enough
temporal depth to train the SpatialTemporalTransformer with
lookback=10 days of surface snapshots.

Target tensor: X_grid = [Samples, Lookback=10, Tenors=5, Deltas=25]
Target labels: y_vrp  = IV(t) - RealizedVol(t+5)  (5-day forward VRP)
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/vrp_daily.log", mode="a"),
    ],
)
logger = logging.getLogger("VRP_Daily")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GRID_DIR = PROJECT_ROOT / "data_lake" / "vrp_history" / "grids"
MASTER_DIR = PROJECT_ROOT / "data_lake" / "vrp_history"


def run_daily_snapshot():
    """Fetch today's IV surface, save as date-stamped grid."""
    from backend.app.ml_platform.models.vol_surface_grid.options_pipeline import (
        fetch_spy_options_chain,
        interpolate_to_grid,
    )

    GRID_DIR.mkdir(parents=True, exist_ok=True)
    today = datetime.now().strftime("%Y-%m-%d")
    grid_path = GRID_DIR / f"iv_grid_{today}.npy"
    raw_path = GRID_DIR / f"iv_surface_{today}.parquet"
    meta_path = GRID_DIR / f"meta_{today}.json"

    if grid_path.exists():
        logger.info("Grid for %s already exists. Skipping.", today)
        return

    try:
        logger.info("Fetching SPY options chain for %s...", today)
        iv_surface = fetch_spy_options_chain()
        logger.info("  %d valid option quotes", len(iv_surface))

        grid = interpolate_to_grid(iv_surface)
        logger.info("  Grid shape: %s  range: [%.4f, %.4f]",
                     grid.shape, grid.min(), grid.max())

        # Save daily artifacts
        np.save(grid_path, grid)
        iv_surface.to_parquet(raw_path)

        # Compute surface statistics
        spot_price = None
        try:
            import yfinance as yf
            spot_price = float(yf.Ticker("SPY").history(period="1d")["Close"].iloc[-1])
        except Exception:
            pass

        meta = {
            "date": today,
            "grid_shape": list(grid.shape),
            "num_options": len(iv_surface),
            "iv_mean": float(np.nanmean(grid)),
            "iv_std": float(np.nanstd(grid)),
            "iv_skew": float(np.nanmean(grid[:, :5]) - np.nanmean(grid[:, -5:])),
            "spot_price": spot_price,
            "expirations_used": sorted(iv_surface["dte"].unique().tolist()),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        logger.info("  Saved: %s (%.1f KB)", grid_path.name,
                     grid_path.stat().st_size / 1024)

    except Exception as e:
        logger.error("FAILED to capture grid for %s: %s", today, e)
        return

    # Rebuild master tensor
    rebuild_master_tensor()


def rebuild_master_tensor():
    """Rebuild the running master tensor from all daily grids.

    Creates:
      - grids_master.npy:  [N_days, 5, 25]  — one grid per day
      - dates_master.json: date labels for each row
      - X_grid.npy:        [Samples, Lookback=10, 5, 25] — sliding windows
      - y_vrp.npy:         [Samples, 5, 25] — placeholder (updated when RV available)
    """
    grid_files = sorted(GRID_DIR.glob("iv_grid_*.npy"))
    if len(grid_files) < 2:
        logger.info("Only %d grids available. Need more days.", len(grid_files))
        return

    # Stack all daily grids
    grids = []
    dates = []
    for gf in grid_files:
        g = np.load(gf)
        if g.shape == (5, 25):
            grids.append(g)
            date_str = gf.stem.replace("iv_grid_", "")
            dates.append(date_str)

    master = np.stack(grids, axis=0)  # [N_days, 5, 25]
    np.save(MASTER_DIR / "grids_master.npy", master)
    with open(MASTER_DIR / "dates_master.json", "w") as f:
        json.dump(dates, f)

    logger.info("Master tensor: %s (%d trading days)", master.shape, len(dates))

    # Build sliding window training tensor (lookback=10)
    lookback = 10
    if len(grids) >= lookback + 5:  # Need 5 forward days for VRP target
        n_samples = len(grids) - lookback - 5 + 1
        X = np.zeros((n_samples, lookback, 5, 25), dtype=np.float32)
        y = np.zeros((n_samples, 5, 25), dtype=np.float32)

        for i in range(n_samples):
            X[i] = master[i:i + lookback]
            # VRP proxy target: IV change over next 5 days
            # (will be replaced with IV - RV when realized vol is computed)
            y[i] = master[i + lookback + 4] - master[i + lookback]

        np.save(MASTER_DIR / "X_grid.npy", X)
        np.save(MASTER_DIR / "y_vrp.npy", y)
        logger.info("Training tensor: X=%s y=%s", X.shape, y.shape)
    else:
        logger.info("Need %d+ days for training windows. Have %d.",
                     lookback + 5, len(grids))


if __name__ == "__main__":
    run_daily_snapshot()
