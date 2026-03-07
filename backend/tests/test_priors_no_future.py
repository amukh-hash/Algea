"""Priors anti-leakage tests — verify build_priors excludes future data.

Tests the strict < asof filtering, sentinel-row immunity, and schema
validation of the distributional priors pipeline.
"""
import pytest
from datetime import date

import numpy as np
import pandas as pd

from algae.data.priors.build import build_priors
from algae.data.priors.validate import PriorsValidationError
from algae.models.foundation.base import StatisticalFallbackProvider, PRIORS_REQUIRED_COLUMNS
from algae.models.foundation.chronos2 import FoundationModelConfig


def _make_canonical(n_days: int = 30) -> pd.DataFrame:
    """Generate deterministic canonical daily data."""
    np.random.seed(123)
    dates = pd.bdate_range("2024-01-01", periods=n_days)
    close = 50 + np.cumsum(np.random.randn(n_days) * 0.3)
    return pd.DataFrame({
        "date": dates,
        "ticker": "AAA",
        "open": close - 0.2,
        "high": close + 0.5,
        "low": close - 0.5,
        "close": close,
        "volume": 500 + np.random.randint(0, 100, n_days),
    })


def test_priors_ignore_future_data():
    """Priors computed with absurd future rows must equal priors without them."""
    provider = StatisticalFallbackProvider()
    base = _make_canonical(25)
    asof = date.fromisoformat("2024-01-31")

    # Add absurd future sentinel rows
    sentinel = pd.DataFrame({
        "date": pd.bdate_range("2024-02-01", periods=5),
        "ticker": "AAA",
        "open": [1e6] * 5,
        "high": [1e6] * 5,
        "low": [1e6] * 5,
        "close": [1e6] * 5,
        "volume": [1e9] * 5,
    })
    full = pd.concat([base, sentinel], ignore_index=True)

    priors_clean = build_priors(base, asof, provider=provider)
    priors_dirty = build_priors(full, asof, provider=provider)

    pd.testing.assert_frame_equal(
        priors_clean.reset_index(drop=True),
        priors_dirty.reset_index(drop=True),
    )


def test_priors_schema_complete():
    """Priors output must contain all required canonical columns."""
    provider = StatisticalFallbackProvider()
    data = _make_canonical(25)
    asof = date.fromisoformat("2024-01-31")
    priors = build_priors(data, asof, provider=provider)

    for col in PRIORS_REQUIRED_COLUMNS:
        assert col in priors.columns, f"Missing required column: {col}"


def test_priors_sigma_positive():
    """Volatility estimates must be strictly positive."""
    provider = StatisticalFallbackProvider()
    data = _make_canonical(25)
    asof = date.fromisoformat("2024-01-31")
    priors = build_priors(data, asof, provider=provider)

    assert (priors["p_sig5"] > 0).all(), "p_sig5 must be positive"
    assert (priors["p_sig10"] > 0).all(), "p_sig10 must be positive"


def test_priors_pdown_bounded():
    """Probability of down must be in [0, 1]."""
    provider = StatisticalFallbackProvider()
    data = _make_canonical(25)
    asof = date.fromisoformat("2024-01-31")
    priors = build_priors(data, asof, provider=provider)

    for col in ["p_pdown5", "p_pdown10"]:
        assert (priors[col] >= 0).all(), f"{col} must be >= 0"
        assert (priors[col] <= 1).all(), f"{col} must be <= 1"
