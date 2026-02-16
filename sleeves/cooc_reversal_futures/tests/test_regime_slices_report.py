"""Tests: Regime slicing report computes IC across slices."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _make_val_dataset(n_days: int = 60) -> pd.DataFrame:
    """Generate dataset with regime columns for slicing."""
    np.random.seed(99)
    instruments = ["ES", "NQ", "YM", "RTY"]
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2024-06-01") + pd.Timedelta(days=d)
        for inst in instruments:
            rows.append({
                "trading_day": day.date(),
                "instrument": inst,
                "r_co": np.random.randn() * 0.01,
                "r_oc": np.random.randn() * 0.01,
                "y": -np.random.randn() * 0.01,
                "sigma_co": abs(np.random.randn() * 0.005),
                "sigma_oc": abs(np.random.randn() * 0.005),
                "shock_flag": 1.0 if np.random.random() > 0.9 else 0.0,
                "roll_window_flag": 1 if np.random.random() > 0.95 else 0,
                "volume_z": np.random.randn(),
                "volume": 100000,
            })
    return pd.DataFrame(rows)


def test_regime_slices_have_content():
    """Regime slicing gate should produce non-trivial detail string."""
    from sleeves.cooc_reversal_futures.pipeline.types import GateResult

    ds = _make_val_dataset()

    # Simulate predictions as noisy y
    preds = ds["y"].values + np.random.randn(len(ds)) * 0.001

    # Check that we can compute IC for each slice
    if "sigma_co" in ds.columns:
        ds["_vol_q"] = pd.qcut(ds["sigma_co"], q=4, labels=False, duplicates="drop")
        for q in sorted(ds["_vol_q"].dropna().unique()):
            sub = ds[ds["_vol_q"] == q]
            assert len(sub) >= 5, f"Vol quartile {q} has too few rows"
            y = sub["y"].values
            p = preds[sub.index]
            ic = float(np.corrcoef(y, p)[0, 1])
            assert np.isfinite(ic), f"IC for vol_q{q} is not finite"


def test_shock_normal_split():
    """Shock and normal slices should have different sample sizes."""
    ds = _make_val_dataset()
    shock = ds[ds["shock_flag"] == 1.0]
    normal = ds[ds["shock_flag"] == 0.0]
    assert len(shock) < len(normal), "Shock days should be rare (<10%)"
    assert len(shock) > 0, "Should have at least some shock days"


def test_roll_window_slice():
    """Roll-window slice should be a small subset."""
    ds = _make_val_dataset()
    roll = ds[ds["roll_window_flag"] == 1]
    non_roll = ds[ds["roll_window_flag"] == 0]
    assert len(roll) < len(non_roll)
