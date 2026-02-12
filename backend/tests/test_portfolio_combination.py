"""
Tests for portfolio combination — weighted returns and NAV reconstruction.
"""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.analysis.sharpe_report import load_daily_data, generate_sharpe_report


class TestPortfolioCombination:
    """Verify the weighted combination and NAV reconstruction are correct."""

    @staticmethod
    def _make_data(N=200, seed=42):
        """Create deterministic synthetic daily data."""
        np.random.seed(seed)
        dates = pd.bdate_range("2023-01-01", periods=N)
        core_pnl = np.random.normal(500, 2000, N)
        vrp_pnl = np.random.normal(200, 1000, N)
        w_vrp = np.clip(np.random.uniform(0.05, 0.25, N), 0, 0.25)
        regimes = np.random.choice(
            ["normal_carry", "caution", "crash_risk"], N, p=[0.6, 0.3, 0.1]
        )
        return pd.DataFrame({
            "date": dates,
            "core_pnl": core_pnl,
            "vrp_pnl": vrp_pnl,
            "w_vrp": w_vrp,
            "regime": regimes,
        })

    def test_weighted_combination(self):
        """port_ret == (1 - w_vrp) * core_ret + w_vrp * vrp_ret."""
        data = self._make_data()
        df = load_daily_data(data)

        expected_port = (1 - df["w_vrp"]) * df["core_ret"] + df["w_vrp"] * df["vrp_ret"]
        np.testing.assert_allclose(
            df["port_ret"].values, expected_port.values, atol=1e-12,
            err_msg="Combined port_ret does not match weighted sum",
        )

    def test_nav_reconstruction(self):
        """NAV from cumulative PnL matches expected values."""
        data = self._make_data(N=50, seed=7)
        initial_capital = 500_000.0
        df = load_daily_data(data, initial_capital=initial_capital)

        # Manually reconstruct core_nav
        expected_core_nav = initial_capital + data["core_pnl"].cumsum()
        # load_daily_data sorts by date and resets index so should match
        np.testing.assert_allclose(
            df["core_nav"].values, expected_core_nav.values, atol=1e-6,
            err_msg="Core NAV reconstruction mismatch",
        )

    def test_returns_roundtrip(self):
        """core_ret * core_nav_prev ≈ core_pnl for all rows."""
        data = self._make_data(N=100, seed=3)
        initial_capital = 1_000_000.0
        df = load_daily_data(data, initial_capital=initial_capital)

        core_nav_prev = df["core_nav"].shift(1)
        core_nav_prev.iloc[0] = initial_capital
        reconstructed_pnl = df["core_ret"] * core_nav_prev

        np.testing.assert_allclose(
            reconstructed_pnl.values, df["core_pnl"].values, atol=1e-6,
            err_msg="Return * lagged NAV should equal PnL",
        )

    def test_generate_report_structure(self):
        """generate_sharpe_report returns all expected top-level keys."""
        data = self._make_data(N=300)
        report = generate_sharpe_report("test_run", data)

        expected_keys = {
            "run_id", "core", "vrp", "combined",
            "correlation", "marginal_sharpe_contribution",
            "information_ratio", "diversification_benefit",
            "regime_breakdown", "rolling_sharpe_summary",
        }
        assert expected_keys == set(report.keys()), \
            f"Missing or extra keys: {expected_keys.symmetric_difference(set(report.keys()))}"

    def test_w_vrp_zero_gives_core_only(self):
        """When w_vrp=0 everywhere, combined should equal core."""
        np.random.seed(55)
        N = 100
        dates = pd.bdate_range("2023-01-01", periods=N)
        core_pnl = np.random.normal(300, 1500, N)
        data = pd.DataFrame({
            "date": dates,
            "core_pnl": core_pnl,
            "vrp_pnl": np.random.normal(100, 500, N),
            "w_vrp": np.zeros(N),
        })
        df = load_daily_data(data)
        np.testing.assert_allclose(
            df["port_ret"].values, df["core_ret"].values, atol=1e-12,
            err_msg="w_vrp=0 should give port_ret == core_ret",
        )
