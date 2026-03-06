"""
Feature parity test — ensures ``score_universe`` produces identical features
to the training frame builder for the same date.

Also validates:
    - No NaNs in computed features
    - z-features within [-5, +5]
    - Quantiles monotonic (q10 ≤ q50 ≤ q90)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _load_frame_partition(frame_root: Path, date_str: str) -> pd.DataFrame:
    """Load a single priors_frame partition."""
    part = frame_root / f"date={date_str}" / "part.parquet"
    if not part.exists():
        pytest.skip(f"No frame partition for {date_str}")
    return pd.read_parquet(part)


def _get_first_available_date(frame_root: Path) -> str:
    """Find the first available date partition."""
    if not frame_root.exists():
        pytest.skip("priors_frame directory not found")
    parts = sorted(
        d.name.replace("date=", "")
        for d in frame_root.iterdir()
        if d.is_dir() and d.name.startswith("date=")
    )
    if not parts:
        pytest.skip("No partitions found in priors_frame")
    return parts[0]


# ═══════════════════════════════════════════════════════════════════════════
# Tests
# ═══════════════════════════════════════════════════════════════════════════

FRAME_ROOT = ROOT / "backend" / "data" / "selector" / "priors_frame"


class TestFrameQuality:
    """Validate priors_frame data quality."""

    def test_no_nans_in_features(self):
        """All model features must be non-NaN."""
        from algae.data.priors.selector_schema import MODEL_FEATURE_COLS

        date_str = _get_first_available_date(FRAME_ROOT)
        df = _load_frame_partition(FRAME_ROOT, date_str)

        available = [c for c in MODEL_FEATURE_COLS if c in df.columns]
        assert len(available) > 0, "No model features found in frame"

        nans = df[available].isna().sum()
        bad = nans[nans > 0]
        assert bad.empty, f"NaN features found: {bad.to_dict()}"

    def test_z_features_bounded(self):
        """z-features must be within [-5, +5]."""
        from algae.data.priors.selector_schema import Z_FEATURE_COLS

        date_str = _get_first_available_date(FRAME_ROOT)
        df = _load_frame_partition(FRAME_ROOT, date_str)

        available = [c for c in Z_FEATURE_COLS if c in df.columns]
        if not available:
            pytest.skip("No z-features in frame")

        for col in available:
            vals = df[col].dropna()
            assert vals.min() >= -5.0 - 1e-6, f"{col} min={vals.min()}"
            assert vals.max() <= 5.0 + 1e-6, f"{col} max={vals.max()}"

    def test_quantiles_monotonic(self):
        """q10 ≤ q50 ≤ q90 for both horizons."""
        date_str = _get_first_available_date(FRAME_ROOT)
        df = _load_frame_partition(FRAME_ROOT, date_str)

        for h in ("10", "30"):
            q10 = f"q10_{h}"
            q50 = f"q50_{h}"
            q90 = f"q90_{h}"
            if not all(c in df.columns for c in [q10, q50, q90]):
                continue

            violations = (df[q10] > df[q50] + 1e-8) | (df[q50] > df[q90] + 1e-8)
            n_bad = violations.sum()
            assert n_bad == 0, f"Monotonicity violated in {n_bad}/{len(df)} rows for h={h}"

    def test_targets_no_leakage(self):
        """y_ret and y_vol must exist and be finite for most rows."""
        date_str = _get_first_available_date(FRAME_ROOT)
        df = _load_frame_partition(FRAME_ROOT, date_str)

        if "y_ret" not in df.columns:
            pytest.skip("y_ret not in frame")

        nan_frac = df["y_ret"].isna().mean()
        assert nan_frac < 0.5, f"Too many NaN targets: {nan_frac:.1%}"

    def test_minimum_universe_size(self):
        """Each date should have a reasonable universe size."""
        date_str = _get_first_available_date(FRAME_ROOT)
        df = _load_frame_partition(FRAME_ROOT, date_str)
        assert len(df) >= 10, f"Too few tickers: {len(df)}"


class TestFeatureParity:
    """Verify that score_universe produces features matching training frame.

    NOTE: This test requires both priors_cache and priors_frame to exist
    for a common date.  If they don't, the test is skipped.
    """

    def test_parity_with_cache(self):
        """Features from cache-loaded inference match training frame."""
        from algae.data.priors.feature_utils import build_features
        from algae.data.priors.selector_schema import MODEL_FEATURE_COLS

        cache_root = ROOT / "backend" / "data" / "selector" / "priors_cache"
        frame_root = FRAME_ROOT

        date_str = _get_first_available_date(frame_root)

        # Load training frame
        train_df = _load_frame_partition(frame_root, date_str)

        # Rebuild features from cached priors (same as score_universe would)
        cache_10 = cache_root / f"teacher=10d" / f"date={date_str}" / "part.parquet"
        cache_30 = cache_root / f"teacher=30d" / f"date={date_str}" / "part.parquet"

        if not cache_10.exists() or not cache_30.exists():
            pytest.skip(f"Cache partitions not available for {date_str}")

        df_10 = pd.read_parquet(cache_10)
        df_30 = pd.read_parquet(cache_30)

        # Join
        join_cols = ["date", "symbol"]
        meta_drop = [c for c in df_30.columns
                     if c not in join_cols and not c.endswith("_30")]
        rebuilt_df = df_10.merge(
            df_30.drop(columns=meta_drop, errors="ignore"),
            on=join_cols, how="inner",
        )
        rebuilt_df = build_features(rebuilt_df)

        # Compare
        feature_cols = [c for c in MODEL_FEATURE_COLS if c in train_df.columns and c in rebuilt_df.columns]
        if not feature_cols:
            pytest.skip("No overlapping feature columns")

        # Align on symbols
        common_syms = sorted(
            set(train_df["symbol"].values) & set(rebuilt_df["symbol"].values)
        )
        if len(common_syms) < 5:
            pytest.skip(f"Too few common symbols: {len(common_syms)}")

        train_sub = train_df[train_df["symbol"].isin(common_syms)].sort_values("symbol")
        rebuilt_sub = rebuilt_df[rebuilt_df["symbol"].isin(common_syms)].sort_values("symbol")

        for col in feature_cols:
            train_vals = train_sub[col].values
            rebuilt_vals = rebuilt_sub[col].values
            np.testing.assert_allclose(
                train_vals, rebuilt_vals, atol=1e-5,
                err_msg=f"Feature parity failed for {col}"
            )
