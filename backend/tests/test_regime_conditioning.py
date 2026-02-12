"""
Tests for regime-conditioned Sharpe — verify filtering and per-regime metrics.
"""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.analysis.sharpe_report import (
    compute_regime_sharpe,
    compute_sharpe,
    load_daily_data,
)


class TestRegimeConditioning:
    """Verify regime subset filtering and per-regime Sharpe."""

    @staticmethod
    def _make_regime_data(seed=42):
        """Create data with deterministic, distinct regime blocks."""
        np.random.seed(seed)
        N = 300
        dates = pd.bdate_range("2023-01-01", periods=N)

        # 150 normal, 100 caution, 50 crash
        regimes = (
            ["normal_carry"] * 150
            + ["caution"] * 100
            + ["crash_risk"] * 50
        )

        core_pnl = np.random.normal(400, 1500, N)
        vrp_pnl = np.random.normal(200, 800, N)
        w_vrp = np.full(N, 0.10)

        return pd.DataFrame({
            "date": dates,
            "core_pnl": core_pnl,
            "vrp_pnl": vrp_pnl,
            "w_vrp": w_vrp,
            "regime": regimes,
        })

    def test_regime_filter_correct_counts(self):
        """Each regime subset has the correct number of observations."""
        data = self._make_regime_data()
        df = load_daily_data(data)
        result = compute_regime_sharpe(df)

        assert result["normal_carry"]["n_obs"] == 150
        assert result["caution"]["n_obs"] == 100
        assert result["crash_risk"]["n_obs"] == 50

    def test_regime_sharpe_matches_manual(self):
        """Per-regime Sharpe matches manual filter-then-compute."""
        data = self._make_regime_data()
        df = load_daily_data(data)
        result = compute_regime_sharpe(df, rf_annual=0.0)

        # Manually compute for normal_carry core
        normal_mask = df["regime"] == "normal_carry"
        manual_sharpe = compute_sharpe(df.loc[normal_mask, "core_ret"])

        assert abs(result["normal_carry"]["core_sharpe"] - manual_sharpe) < 1e-5

    def test_crash_risk_sharpe_manual(self):
        """Crash risk subset Sharpe matches manual computation."""
        data = self._make_regime_data()
        df = load_daily_data(data)
        result = compute_regime_sharpe(df, rf_annual=0.0)

        crash_mask = df["regime"] == "crash_risk"
        manual = compute_sharpe(df.loc[crash_mask, "vrp_ret"])
        assert abs(result["crash_risk"]["vrp_sharpe"] - manual) < 1e-5

    def test_small_regime_returns_none(self):
        """Regime with < 30 obs should report Sharpe as None."""
        np.random.seed(99)
        N = 50
        dates = pd.bdate_range("2023-01-01", periods=N)

        # Only 10 crash days → too few for Sharpe
        regimes = ["normal_carry"] * 40 + ["crash_risk"] * 10

        data = pd.DataFrame({
            "date": dates,
            "core_pnl": np.random.normal(300, 1000, N),
            "vrp_pnl": np.random.normal(100, 500, N),
            "w_vrp": np.full(N, 0.10),
            "regime": regimes,
        })
        df = load_daily_data(data)
        result = compute_regime_sharpe(df)

        # crash_risk has only 10 obs → Sharpe should be None
        assert result["crash_risk"]["core_sharpe"] is None
        # normal_carry has 40 obs → Sharpe should be a float
        assert isinstance(result["normal_carry"]["core_sharpe"], float)

    def test_no_regime_column_returns_empty(self):
        """When regime is absent, regime Sharpe returns empty dict."""
        np.random.seed(7)
        N = 100
        dates = pd.bdate_range("2023-01-01", periods=N)
        data = pd.DataFrame({
            "date": dates,
            "core_pnl": np.random.normal(300, 1000, N),
            "vrp_pnl": np.random.normal(100, 500, N),
            "w_vrp": np.full(N, 0.10),
        })
        df = load_daily_data(data)
        result = compute_regime_sharpe(df)
        assert result == {}
