"""
Edge-case tests for Sharpe analysis — zero vol, constant returns, small samples.
"""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.analysis.sharpe_report import compute_sharpe, MIN_OBS


class TestEdgeCases:
    """Verify correct handling of degenerate inputs."""

    def test_zero_vol_returns_zero_sharpe(self):
        """Constant returns → zero vol → Sharpe = 0."""
        returns = pd.Series([0.001] * 100)
        sharpe = compute_sharpe(returns, rf_annual=0.0)
        assert sharpe == 0.0, f"Expected Sharpe=0 for constant returns, got {sharpe}"

    def test_constant_zero_returns(self):
        """All-zero returns → vol = 0 → Sharpe = 0."""
        returns = pd.Series([0.0] * 100)
        sharpe = compute_sharpe(returns, rf_annual=0.0)
        assert sharpe == 0.0

    def test_constant_returns_nonzero_rf(self):
        """Constant returns with rf ≠ 0 → still zero vol → Sharpe = 0."""
        returns = pd.Series([0.002] * 100)
        sharpe = compute_sharpe(returns, rf_annual=0.05)
        # excess = 0.002 - 0.05/252 ≈ 0.002 - 0.000198 = constant
        # std of constant = 0 → Sharpe = 0
        assert sharpe == 0.0

    def test_small_sample_rejection(self):
        """Fewer than MIN_OBS observations → ValueError."""
        returns = pd.Series([0.001] * (MIN_OBS - 1))
        with pytest.raises(ValueError, match="Insufficient data"):
            compute_sharpe(returns)

    def test_exactly_min_obs(self):
        """Exactly MIN_OBS observations → should NOT raise."""
        np.random.seed(12)
        returns = pd.Series(np.random.normal(0.0003, 0.01, MIN_OBS))
        sharpe = compute_sharpe(returns)
        assert isinstance(sharpe, float)

    def test_single_observation_raises(self):
        """Single observation → ValueError."""
        with pytest.raises(ValueError):
            compute_sharpe(pd.Series([0.01]))

    def test_empty_series_raises(self):
        """Empty series → ValueError."""
        with pytest.raises(ValueError):
            compute_sharpe(pd.Series(dtype=float))

    def test_very_large_returns(self):
        """Very large returns should not cause overflow."""
        np.random.seed(77)
        returns = pd.Series(np.random.normal(1.0, 0.5, 100))
        sharpe = compute_sharpe(returns)
        assert np.isfinite(sharpe), "Sharpe should be finite for large but valid returns"

    def test_very_small_returns(self):
        """Very small (near-zero) returns should produce finite Sharpe."""
        np.random.seed(77)
        returns = pd.Series(np.random.normal(1e-8, 1e-9, 100))
        sharpe = compute_sharpe(returns)
        assert np.isfinite(sharpe)
