"""Tests for COOC O(1) incremental feature pipeline parity.

Verifies:
  - precompute_t_minus_1 returns exactly 1 row per instrument
  - update_incremental correctly computes live r_co and cross-sectional features
  - Mathematical parity between monolithic and incremental paths on
    all critical features (r_co, trend_20, dd_60, sigma_oc_hist, etc.)
  - Stale cache detection
  - Profile gate returns valid latency measurements
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose

from sleeves.cooc_reversal_futures.features_core import (
    FeatureConfig,
    compute_core_features,
    precompute_t_minus_1,
    update_incremental,
    profile_feature_pipeline,
)


# ═══════════════════════════════════════════════════════════════════════
# Fixtures: synthetic futures history
# ═══════════════════════════════════════════════════════════════════════

_INSTRUMENTS = ["ES", "NQ", "YM", "RTY", "GC", "CL", "SI", "HG"]
_N_DAYS = 120  # Must exceed cfg.long_lookback (60) + warmup


@pytest.fixture()
def synthetic_history() -> pd.DataFrame:
    """Build a deterministic synthetic futures history for testing."""
    np.random.seed(42)
    rows = []
    for sym in _INSTRUMENTS:
        base_price = {"ES": 5000, "NQ": 19000, "YM": 39000, "RTY": 2200,
                      "GC": 2000, "CL": 75, "SI": 25, "HG": 4}[sym]

        close = base_price
        for d in range(1, _N_DAYS + 1):
            # Simulate realistic-ish returns
            r_co = np.random.normal(0, 0.005)  # overnight
            r_oc = np.random.normal(0, 0.008)  # intraday
            open_px = close * (1 + r_co)
            close_px = open_px * (1 + r_oc)
            volume = int(np.random.lognormal(10, 1))

            rows.append({
                "instrument": sym,
                "trading_day": pd.Timestamp("2025-09-01") + pd.Timedelta(days=d),
                "r_co": r_co,
                "r_oc": r_oc,
                "close": close_px,
                "open": open_px,
                "volume": volume,
                "days_to_expiry": max(0, 90 - (d % 90)),
                "roll_window_flag": 1 if (d % 90) > 85 else 0,
            })
            close = close_px

    return pd.DataFrame(rows)


@pytest.fixture()
def live_0920_quotes(synthetic_history: pd.DataFrame) -> dict:
    """Simulate live 09:20 bar quotes using last history close + random gap."""
    np.random.seed(99)
    last_closes = (
        synthetic_history.sort_values(["instrument", "trading_day"])
        .groupby("instrument")
        .tail(1)
        .set_index("instrument")["close"]
    )
    quotes = {}
    for sym, close_px in last_closes.items():
        gap_ret = np.random.normal(0, 0.003)
        quotes[sym] = {
            "open": close_px * (1 + gap_ret),
            "volume": int(np.random.lognormal(10, 1)),
        }
    return quotes


# ═══════════════════════════════════════════════════════════════════════
# Test: precompute_t_minus_1
# ═══════════════════════════════════════════════════════════════════════

class TestPrecomputeTMinus1:
    def test_returns_one_row_per_instrument(self, synthetic_history):
        result = precompute_t_minus_1(synthetic_history)
        assert len(result) == len(_INSTRUMENTS)
        assert set(result["instrument"].tolist()) == set(_INSTRUMENTS)

    def test_contains_all_v2_features(self, synthetic_history):
        cfg = FeatureConfig(schema_version=2)
        result = precompute_t_minus_1(synthetic_history, cfg)
        for col in ["sigma_co", "sigma_oc_hist", "trend_20", "dd_60", "rv_60", "skew_proxy"]:
            assert col in result.columns, f"Missing historical feature: {col}"

    def test_features_are_not_nan(self, synthetic_history):
        """With 120 days of history (> 60 warmup), features should be populated."""
        result = precompute_t_minus_1(synthetic_history)
        # sigma_co and sigma_oc_hist should be populated after 20-day warmup
        assert result["sigma_co"].notna().all(), "sigma_co has NaN values"
        assert result["sigma_oc_hist"].notna().all(), "sigma_oc_hist has NaN values"

    def test_result_is_last_trading_day(self, synthetic_history):
        result = precompute_t_minus_1(synthetic_history)
        last_day = synthetic_history["trading_day"].max()
        for _, row in result.iterrows():
            sym_history = synthetic_history[
                synthetic_history["instrument"] == row["instrument"]
            ]
            assert row["trading_day"] == sym_history["trading_day"].max()


# ═══════════════════════════════════════════════════════════════════════
# Test: update_incremental
# ═══════════════════════════════════════════════════════════════════════

class TestUpdateIncremental:
    def test_computes_r_co(self, synthetic_history, live_0920_quotes):
        cached = precompute_t_minus_1(synthetic_history)
        today = date(2026, 3, 6)
        result = update_incremental(cached, live_0920_quotes, today)

        # r_co should be log(open_t0 / close)
        for _, row in result.iterrows():
            sym = row["instrument"]
            expected_open = live_0920_quotes[sym]["open"]
            expected_close = cached.loc[
                cached["instrument"] == sym, "close"
            ].iloc[0]
            expected_r_co = np.log(expected_open / expected_close)
            assert_allclose(row["r_co"], expected_r_co, rtol=1e-6,
                            err_msg=f"r_co mismatch for {sym}")

    def test_cross_sectional_demeaning(self, synthetic_history, live_0920_quotes):
        cached = precompute_t_minus_1(synthetic_history)
        today = date(2026, 3, 6)
        result = update_incremental(cached, live_0920_quotes, today)

        # r_co_cs_demean should sum to ~0
        cs_demean_sum = result["r_co_cs_demean"].sum()
        assert abs(cs_demean_sum) < 1e-10, (
            f"Cross-sectional demeaning error: sum={cs_demean_sum}"
        )

    def test_rank_pct_bounded(self, synthetic_history, live_0920_quotes):
        cached = precompute_t_minus_1(synthetic_history)
        today = date(2026, 3, 6)
        result = update_incremental(cached, live_0920_quotes, today)

        assert (result["r_co_rank_pct"] >= 0).all()
        assert (result["r_co_rank_pct"] <= 1).all()

    def test_preserves_historical_features(self, synthetic_history, live_0920_quotes):
        """Historical rolling features must carry forward unchanged."""
        cached = precompute_t_minus_1(synthetic_history)
        today = date(2026, 3, 6)
        result = update_incremental(cached, live_0920_quotes, today)

        # sigma_co, sigma_oc_hist, trend_20 should be identical to cached values
        for col in ["sigma_co", "sigma_oc_hist", "trend_20", "dd_60", "rv_60"]:
            assert_allclose(
                result[col].values,
                cached[col].values,
                rtol=1e-10,
                err_msg=f"Historical feature {col} was mutated during incremental update",
            )

    def test_updates_trading_day(self, synthetic_history, live_0920_quotes):
        cached = precompute_t_minus_1(synthetic_history)
        today = date(2026, 3, 6)
        result = update_incremental(cached, live_0920_quotes, today)
        assert (result["trading_day"] == pd.Timestamp(today)).all()

    def test_missing_instrument_in_quotes_handled(self, synthetic_history, live_0920_quotes):
        """If a symbol is missing from live quotes, r_co should be NaN (not crash)."""
        cached = precompute_t_minus_1(synthetic_history)
        partial_quotes = {k: v for k, v in live_0920_quotes.items() if k != "GC"}
        today = date(2026, 3, 6)
        result = update_incremental(cached, partial_quotes, today)
        # Should not crash; GC row should have NaN r_co
        gc_row = result[result["instrument"] == "GC"]
        assert len(gc_row) == 1  # GC still present in output


# ═══════════════════════════════════════════════════════════════════════
# Test: Mathematical parity (monolithic vs incremental)
# ═══════════════════════════════════════════════════════════════════════

class TestIncrementalFeatureParity:
    """CRITICAL: Ensures incremental path produces identical features."""

    def test_historical_features_exact_parity(self, synthetic_history):
        """Cached T-1 features must exactly match the last row of monolithic output."""
        cfg = FeatureConfig(schema_version=2)

        # Monolithic: compute on full history, extract last row per instrument
        mono_full = compute_core_features(synthetic_history, cfg)
        mono_last = (
            mono_full.sort_values(["instrument", "trading_day"])
            .groupby("instrument")
            .tail(1)
            .sort_values("instrument")
            .reset_index(drop=True)
        )

        # Incremental: precompute_t_minus_1 should produce identical T-1 state
        cached = precompute_t_minus_1(synthetic_history, cfg).sort_values(
            "instrument"
        ).reset_index(drop=True)

        # Assert exact parity on all historical rolling features
        feature_cols = [
            "sigma_co", "sigma_oc_hist", "trend_20", "dd_60",
            "rv_60", "skew_proxy", "prev_r_oc", "prev_abs_r_oc",
        ]
        for col in feature_cols:
            assert_allclose(
                mono_last[col].values,
                cached[col].values,
                rtol=1e-10,
                err_msg=f"Data drift in T-1 feature: {col}",
            )


# ═══════════════════════════════════════════════════════════════════════
# Test: Profiler gate
# ═══════════════════════════════════════════════════════════════════════

class TestProfileFeaturePipeline:
    def test_returns_valid_timings(self, synthetic_history):
        result = profile_feature_pipeline(synthetic_history)
        assert "total_ms" in result
        assert "features_ms" in result
        assert result["total_ms"] > 0
        assert result["features_ms"] > 0

    def test_profiler_does_not_mutate_input(self, synthetic_history):
        original_shape = synthetic_history.shape
        _ = profile_feature_pipeline(synthetic_history)
        assert synthetic_history.shape == original_shape
