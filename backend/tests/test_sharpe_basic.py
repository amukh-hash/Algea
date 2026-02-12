"""
Tests for basic Sharpe computation — deterministic synthetic returns.
"""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.analysis.sharpe_report import compute_sharpe, TRADING_DAYS_PER_YEAR


class TestSharpeBasic:
    """Verify analytic Sharpe matches expected value on synthetic data."""

    def test_sharpe_deterministic(self):
        """Known distribution → compute expected Sharpe analytically."""
        np.random.seed(42)
        N = 500
        daily_mu = 0.0004  # ~10% annualised
        daily_sigma = 0.01  # ~15.9% annualised
        returns = pd.Series(np.random.normal(daily_mu, daily_sigma, N))

        sharpe = compute_sharpe(returns, rf_annual=0.0)

        # Manual computation
        excess = returns - 0.0
        expected = float(np.sqrt(252) * excess.mean() / excess.std(ddof=1))

        assert abs(sharpe - expected) < 1e-10, \
            f"Sharpe mismatch: {sharpe} vs {expected}"

    def test_sharpe_with_risk_free(self):
        """Sharpe with non-zero risk-free rate."""
        np.random.seed(123)
        N = 300
        returns = pd.Series(np.random.normal(0.0005, 0.012, N))
        rf = 0.04  # 4% annual

        sharpe = compute_sharpe(returns, rf_annual=rf)

        rf_daily = rf / 252
        excess = returns - rf_daily
        expected = float(np.sqrt(252) * excess.mean() / excess.std(ddof=1))

        assert abs(sharpe - expected) < 1e-10

    def test_sharpe_zero_rf_default(self):
        """Default rf=0 works correctly."""
        np.random.seed(7)
        returns = pd.Series(np.random.normal(0.0003, 0.008, 100))

        sharpe_explicit = compute_sharpe(returns, rf_annual=0.0)
        sharpe_default = compute_sharpe(returns)

        assert sharpe_explicit == sharpe_default

    def test_positive_returns_positive_sharpe(self):
        """All-positive mean returns → positive Sharpe."""
        np.random.seed(99)
        returns = pd.Series(np.random.normal(0.001, 0.005, 200))
        sharpe = compute_sharpe(returns)
        assert sharpe > 0, f"Expected positive Sharpe, got {sharpe}"

    def test_negative_returns_negative_sharpe(self):
        """Strongly negative mean returns → negative Sharpe."""
        np.random.seed(99)
        returns = pd.Series(np.random.normal(-0.005, 0.005, 200))
        sharpe = compute_sharpe(returns)
        assert sharpe < 0, f"Expected negative Sharpe, got {sharpe}"
