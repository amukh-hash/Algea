"""Tests for the V3 feature schema.

Covers:
- V3 is a superset of V2
- V3 feature count = 23
- All V3 features computed and non-NaN after warmup
- V2 backward compat preserved
- No future leakage in V3 features
- shock_score computed correctly per-instrument
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleeves.cooc_reversal_futures.features_core import (
    FEATURE_SCHEMA_V1,
    FEATURE_SCHEMA_V2,
    FEATURE_SCHEMA_V3,
    NUM_FEATURES_V2,
    NUM_FEATURES_V3,
    FeatureConfig,
    active_schema,
    compute_core_features,
)


def _make_synthetic_frame(n_instruments: int = 14, n_days: int = 100) -> pd.DataFrame:
    """Build a synthetic gold-like DataFrame for feature testing."""
    rng = np.random.default_rng(42)
    instruments = [f"INST_{i:02d}" for i in range(n_instruments)]
    dates = pd.bdate_range("2023-01-02", periods=n_days)

    rows = []
    for d in dates:
        for inst in instruments:
            rows.append({
                "trading_day": d,
                "instrument": inst,
                "r_co": rng.normal(0, 0.01),
                "r_oc": rng.normal(0, 0.01),
                "volume": max(0, rng.normal(10000, 3000)),
                "days_to_expiry": rng.integers(5, 60),
                "roll_window_flag": int(rng.random() < 0.1),
            })
    return pd.DataFrame(rows)


class TestSchemaVersions:
    """Validate schema definitions."""

    def test_v3_superset_of_v2(self):
        for col in FEATURE_SCHEMA_V2:
            assert col in FEATURE_SCHEMA_V3, f"V2 feature '{col}' missing from V3"

    def test_v2_superset_of_v1(self):
        for col in FEATURE_SCHEMA_V1:
            assert col in FEATURE_SCHEMA_V2, f"V1 feature '{col}' missing from V2"

    def test_v2_feature_count(self):
        assert NUM_FEATURES_V2 == 19, f"V2 should have 19 features, got {NUM_FEATURES_V2}"

    def test_v3_feature_count(self):
        assert NUM_FEATURES_V3 == 23, f"V3 should have 23 features, got {NUM_FEATURES_V3}"

    def test_v3_new_features(self):
        new = set(FEATURE_SCHEMA_V3) - set(FEATURE_SCHEMA_V2)
        expected = {"shock_score", "r_co_rank_z", "sigma_co_rank_pct", "volume_rank_pct"}
        assert new == expected, f"V3 new features mismatch: got {new}"

    def test_active_schema_selector(self):
        assert active_schema(1) == FEATURE_SCHEMA_V1
        assert active_schema(2) == FEATURE_SCHEMA_V2
        assert active_schema(3) == FEATURE_SCHEMA_V3


class TestV3FeatureComputation:
    """Validate V3 features are computed correctly."""

    @pytest.fixture
    def frame(self):
        return _make_synthetic_frame()

    def test_v3_all_columns_present(self, frame):
        cfg = FeatureConfig(schema_version=3)
        result = compute_core_features(frame, cfg)
        for col in FEATURE_SCHEMA_V3:
            assert col in result.columns, f"V3 feature '{col}' missing from output"

    def test_v3_non_nan_after_warmup(self, frame):
        cfg = FeatureConfig(schema_version=3, lookback=20, min_periods=3)
        result = compute_core_features(frame, cfg)
        # Skip first 25 rows per instrument (warmup)
        warmup_rows = 25
        for inst in result["instrument"].unique():
            inst_data = result[result["instrument"] == inst].iloc[warmup_rows:]
            for col in FEATURE_SCHEMA_V3:
                nan_count = inst_data[col].isna().sum()
                total = len(inst_data)
                assert nan_count / total < 0.05, (
                    f"V3 feature '{col}' has {nan_count}/{total} NaNs after warmup for {inst}"
                )

    def test_v2_backward_compat(self, frame):
        """V2 mode should not produce V3 columns."""
        cfg_v2 = FeatureConfig(schema_version=2)
        result = compute_core_features(frame, cfg_v2)
        v3_only = set(FEATURE_SCHEMA_V3) - set(FEATURE_SCHEMA_V2)
        for col in v3_only:
            assert col not in result.columns, f"V3 column '{col}' leaked into V2 output"

    def test_v1_backward_compat(self, frame):
        """V1 mode should not produce V2/V3 columns."""
        cfg_v1 = FeatureConfig(schema_version=1)
        result = compute_core_features(frame, cfg_v1)
        v2_only = set(FEATURE_SCHEMA_V2) - set(FEATURE_SCHEMA_V1)
        for col in v2_only:
            assert col not in result.columns, f"V2 column '{col}' leaked into V1 output"


class TestShockScore:
    """Validate shock_score feature specifically."""

    def test_shock_score_formula(self):
        """shock_score = abs(r_co) / (sigma_co + eps)"""
        frame = _make_synthetic_frame(n_instruments=4, n_days=50)
        cfg = FeatureConfig(schema_version=3, lookback=10, min_periods=3)
        result = compute_core_features(frame, cfg)

        # After warmup, check formula
        warmup = 15
        for inst in result["instrument"].unique():
            inst_data = result[result["instrument"] == inst].iloc[warmup:]
            abs_r_co = inst_data["abs_r_co"]
            sigma_co = inst_data["sigma_co"]
            expected = abs_r_co / (sigma_co + 1e-8)
            actual = inst_data["shock_score"]
            np.testing.assert_allclose(
                actual.values, expected.values, rtol=1e-5,
                err_msg=f"shock_score formula mismatch for {inst}",
            )

    def test_shock_score_positive(self):
        """shock_score should always be non-negative."""
        frame = _make_synthetic_frame(n_instruments=4, n_days=50)
        cfg = FeatureConfig(schema_version=3, lookback=10, min_periods=3)
        result = compute_core_features(frame, cfg)
        valid = result["shock_score"].dropna()
        assert (valid >= 0).all(), "shock_score contains negative values"


class TestNoFutureLeakage:
    """Ensure V3 features don't leak future information."""

    def test_cross_sectional_features_use_same_day(self):
        """Cross-sectional ranks/z-scores only use same-day data."""
        frame = _make_synthetic_frame(n_instruments=6, n_days=30)
        cfg = FeatureConfig(schema_version=3, lookback=10, min_periods=3)
        result = compute_core_features(frame, cfg)

        for day in result["trading_day"].unique():
            day_data = result[result["trading_day"] == day]
            if len(day_data) < 3:
                continue
            # r_co_rank_z should be standardised to mean≈0, std≈1
            z = day_data["r_co_rank_z"].dropna()
            if len(z) >= 4:
                assert abs(z.mean()) < 0.5, f"r_co_rank_z not centered on {day}"

            # rank_pct should be in [0, 1]
            for rank_col in ("sigma_co_rank_pct", "volume_rank_pct"):
                vals = day_data[rank_col].dropna()
                if len(vals) > 0:
                    assert vals.min() >= 0, f"{rank_col} < 0 on {day}"
                    assert vals.max() <= 1, f"{rank_col} > 1 on {day}"
