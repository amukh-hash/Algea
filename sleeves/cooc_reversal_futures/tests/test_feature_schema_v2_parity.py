"""Tests: V2 feature schema parity between training and runtime."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleeves.cooc_reversal_futures.features_core import (
    FEATURE_SCHEMA,
    FEATURE_SCHEMA_V1,
    FEATURE_SCHEMA_V2,
    FeatureConfig,
    compute_core_features,
)


def _make_gold_frame(n_days: int = 100, instruments: tuple = ("ES", "NQ", "YM", "RTY")) -> pd.DataFrame:
    """Generate a synthetic gold-like frame."""
    np.random.seed(42)
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2024-01-02") + pd.Timedelta(days=d)
        for inst in instruments:
            rows.append({
                "trading_day": day.date(),
                "instrument": inst,
                "r_co": np.random.randn() * 0.01,
                "r_oc": np.random.randn() * 0.01,
                "volume": max(1, int(100_000 + np.random.randn() * 20_000)),
                "days_to_expiry": max(0, 60 - d % 90),
                "roll_window_flag": 1 if (60 - d % 90) <= 3 else 0,
                "close": 5000 + np.random.randn() * 100,
                "multiplier": 50.0,
            })
    return pd.DataFrame(rows)


def test_v2_superset_of_v1():
    """V2 must contain all V1 features."""
    for f in FEATURE_SCHEMA_V1:
        assert f in FEATURE_SCHEMA_V2, f"V1 feature '{f}' missing from V2"


def test_v2_feature_count():
    """V2 should have 19 features."""
    assert len(FEATURE_SCHEMA_V2) == 19


def test_v2_features_computed():
    """All 19 V2 features must be present in output."""
    df = _make_gold_frame()
    cfg = FeatureConfig(schema_version=2)
    out = compute_core_features(df, cfg=cfg)
    for f in FEATURE_SCHEMA_V2:
        assert f in out.columns, f"Feature '{f}' not computed"
        assert out[f].notna().sum() > 0, f"Feature '{f}' is all NaN"


def test_v1_backward_compat():
    """V1 schema mode should produce exactly V1 features (not V2 extras)."""
    df = _make_gold_frame()
    cfg = FeatureConfig(schema_version=1)
    out = compute_core_features(df, cfg=cfg)
    for f in FEATURE_SCHEMA_V1:
        assert f in out.columns
    # V2-only features should NOT be present
    v2_only = set(FEATURE_SCHEMA_V2) - set(FEATURE_SCHEMA_V1)
    for f in v2_only:
        assert f not in out.columns, f"V2-only feature '{f}' present in V1 mode"


def test_schema_default_is_v2():
    """Default FEATURE_SCHEMA should point to V2."""
    assert FEATURE_SCHEMA is FEATURE_SCHEMA_V2


def test_no_future_leakage_in_v2():
    """V2 features must be strictly causal (no look-ahead).

    We check that adding future data doesn't change past feature values.
    """
    base = _make_gold_frame(n_days=50)
    extended = _make_gold_frame(n_days=100)

    cfg = FeatureConfig(schema_version=2)
    out_base = compute_core_features(base, cfg=cfg)
    out_ext = compute_core_features(extended, cfg=cfg)

    # Match on (instrument, trading_day) for the first 50 days
    base_days = set(base["trading_day"].unique())
    out_base_sub = out_base[out_base["trading_day"].isin(base_days)].sort_values(
        ["instrument", "trading_day"]
    ).reset_index(drop=True)
    out_ext_sub = out_ext[out_ext["trading_day"].isin(base_days)].sort_values(
        ["instrument", "trading_day"]
    ).reset_index(drop=True)

    for f in FEATURE_SCHEMA_V2:
        vals_base = out_base_sub[f].fillna(-999).values
        vals_ext = out_ext_sub[f].fillna(-999).values
        np.testing.assert_array_almost_equal(
            vals_base, vals_ext, decimal=10,
            err_msg=f"Feature '{f}' changes when future data is added — leakage!",
        )
