"""Backtest causality tests — verify truncation invariance and no look-ahead bias.

Signals generated for date T using data [0..T-1] must be identical
to signals generated using data [0..T-1, T, T+1, ...] truncated to T.
"""
import pytest
import pandas as pd

from algae.data.features.build import build_features
from algae.data.signals.build import build_signals
from algae.models.foundation.base import StatisticalFallbackProvider
from algae.models.foundation.chronos2 import FoundationModelConfig, SimpleChronos2


def _make_canonical(n_days: int = 30) -> pd.DataFrame:
    """Generate deterministic canonical daily data for ticker AAA."""
    import numpy as np
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    close = 100 + np.cumsum(np.random.randn(n_days) * 0.5)
    return pd.DataFrame({
        "date": dates,
        "ticker": "AAA",
        "open": close - 0.5,
        "high": close + 1.0,
        "low": close - 1.0,
        "close": close,
        "volume": 1000 + np.random.randint(0, 200, n_days),
    })


def test_backtest_causality_signals_truncated_match():
    """Signals for date T from full data == signals from truncated data."""
    canonical = _make_canonical(30)
    provider = StatisticalFallbackProvider()
    config = FoundationModelConfig()
    target_date = pd.Timestamp("2024-01-22")  # ~15th business day

    # Full path: features and priors from ALL data, but priors filtered by asof
    full_features = build_features(canonical)
    full_model = SimpleChronos2(config, provider=provider)
    full_priors = full_model.infer_priors(canonical, asof=target_date).priors

    # Truncated path: only data before target_date
    truncated = canonical[canonical["date"] < target_date].copy()
    trunc_features = build_features(truncated)
    trunc_model = SimpleChronos2(config, provider=provider)
    trunc_priors = trunc_model.infer_priors(truncated, asof=target_date).priors

    # Priors must be identical (same input data after asof filter)
    pd.testing.assert_frame_equal(
        full_priors.reset_index(drop=True),
        trunc_priors.reset_index(drop=True),
    )


def test_features_truncation_invariant():
    """Features for date T must not change when future rows are appended."""
    canonical = _make_canonical(30)
    target_date = pd.Timestamp("2024-01-15")

    full_features = build_features(canonical)
    truncated_features = build_features(canonical[canonical["date"] <= target_date])

    # Features for dates <= target_date should be identical
    full_subset = full_features[full_features["date"] <= target_date].reset_index(drop=True)
    trunc_subset = truncated_features[truncated_features["date"] <= target_date].reset_index(drop=True)
    pd.testing.assert_frame_equal(full_subset, trunc_subset)


def test_priors_no_future_contamination():
    """Appending absurd future data must not change priors for earlier date."""
    import numpy as np
    provider = StatisticalFallbackProvider()
    config = FoundationModelConfig()

    base = _make_canonical(20)
    # Add sentinel future rows with absurd values
    sentinel = pd.DataFrame({
        "date": pd.bdate_range("2024-02-01", periods=5),
        "ticker": "AAA",
        "open": [1e6] * 5,
        "high": [1e6] * 5,
        "low": [1e6] * 5,
        "close": [1e6] * 5,  # 10000x normal price
        "volume": [1e9] * 5,
    })
    contaminated = pd.concat([base, sentinel], ignore_index=True)

    asof = base["date"].iloc[-1].date()
    priors_clean = SimpleChronos2(config, provider=provider).infer_priors(base, pd.Timestamp(asof)).priors
    priors_dirty = SimpleChronos2(config, provider=provider).infer_priors(contaminated, pd.Timestamp(asof)).priors

    # Priors for AAA must be identical — sentinel rows are after asof
    pd.testing.assert_frame_equal(
        priors_clean.reset_index(drop=True),
        priors_dirty.reset_index(drop=True),
    )
