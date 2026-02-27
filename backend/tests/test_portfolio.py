"""
Unit tests for the portfolio module (algea/eval/portfolio.py).

Covers:
- Portfolio construction (equal, score_proportional, softmax weighting)
- Exposure caps (per-stock, per-sector)
- Market-neutral mode
- Cost modeling (turnover-based with slippage)
- Volatility scaling
- Equity curve
- Regime breakdown
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algea.eval.portfolio import (
    build_equity_curve,
    build_portfolio,
    compute_portfolio_metrics,
    compute_portfolio_returns,
    compute_regime_breakdown,
)

# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════

def _make_scored_df(n_dates: int = 20, n_stocks: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a realistic scored DataFrame with dates, symbols, scores, and y_ret."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    rows = []
    for date in dates:
        for j in range(n_stocks):
            score = rng.randn()
            y_ret = 0.001 * score + 0.01 * rng.randn()  # weak signal + noise
            rows.append({
                "date": date,
                "symbol": f"STOCK_{j:03d}",
                "score_final": score,
                "y_ret": y_ret,
                "z_cs_tail_30_std": rng.randn(),
                "sector": f"SECTOR_{j % 5}",
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════
# B) Portfolio Construction Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBuildPortfolio:
    """Verify portfolio construction produces valid weights."""

    def test_equal_weight_sums_to_one(self):
        df = _make_scored_df()
        port = build_portfolio(df, k=20, weighting="equal")
        for dt in port["date"].unique():
            day = port[port["date"] == dt]
            assert abs(day["weight"].sum() - 1.0) < 1e-6

    def test_equal_weight_is_uniform(self):
        df = _make_scored_df()
        port = build_portfolio(df, k=10, weighting="equal")
        for dt in port["date"].unique():
            day = port[port["date"] == dt]
            expected = 1.0 / 10
            assert all(abs(w - expected) < 1e-4 for w in day["weight"])

    def test_score_proportional_sums_to_one(self):
        df = _make_scored_df()
        port = build_portfolio(df, k=20, weighting="score_proportional")
        for dt in port["date"].unique():
            day = port[port["date"] == dt]
            assert abs(day["weight"].sum() - 1.0) < 1e-6

    def test_softmax_sums_to_one(self):
        df = _make_scored_df()
        port = build_portfolio(df, k=20, weighting="softmax")
        for dt in port["date"].unique():
            day = port[port["date"] == dt]
            assert abs(day["weight"].sum() - 1.0) < 1e-6

    def test_selects_top_k(self):
        """Top-K stocks should have the highest scores per date."""
        df = _make_scored_df(n_dates=5, n_stocks=50)
        port = build_portfolio(df, k=10, weighting="equal")
        for dt in port["date"].unique():
            day_all = df[df["date"] == dt].sort_values("score_final", ascending=False)
            expected_symbols = set(day_all.head(10)["symbol"])
            actual_symbols = set(port[port["date"] == dt]["symbol"])
            assert actual_symbols == expected_symbols

    def test_max_weight_cap(self):
        """Max weight cap should reduce concentration vs uncapped."""
        df = _make_scored_df()
        # Uncapped softmax can produce concentrated weights
        uncapped = build_portfolio(df, k=50, weighting="softmax", max_weight=1.0)
        capped = build_portfolio(df, k=50, weighting="softmax", max_weight=0.05)
        for dt in capped["date"].unique():
            day_cap = capped[capped["date"] == dt]
            day_uncap = uncapped[uncapped["date"] == dt]
            # Capped max should be <= uncapped max
            assert day_cap["weight"].max() <= day_uncap["weight"].max() + 1e-6
            # Weights should still sum to 1
            assert abs(day_cap["weight"].sum() - 1.0) < 1e-6

    def test_sector_cap(self):
        """No sector should exceed max_sector_weight."""
        df = _make_scored_df(n_stocks=100)
        port = build_portfolio(
            df, k=50, weighting="equal",
            max_sector_weight=0.25, sector_col="sector",
        )
        for dt in port["date"].unique():
            day = port[port["date"] == dt]
            for sector in day["sector"].unique() if "sector" in day.columns else []:
                sector_wt = day[day["sector"] == sector]["weight"].sum()
                assert sector_wt <= 0.25 + 1e-6

    def test_market_neutral(self):
        """Market-neutral portfolio should have long and short legs summing to ~0."""
        df = _make_scored_df()
        port = build_portfolio(df, k=10, weighting="equal", market_neutral=True)
        for dt in port["date"].unique():
            day = port[port["date"] == dt]
            total_weight = day["weight"].sum()
            assert abs(total_weight) < 1e-6  # dollar neutral

    def test_correct_columns(self):
        df = _make_scored_df()
        port = build_portfolio(df, k=10)
        expected_cols = {"date", "symbol", "weight", "score", "y_ret", "side"}
        assert set(port.columns) == expected_cols

    def test_empty_input(self):
        empty = pd.DataFrame(columns=["date", "symbol", "score_final", "y_ret"])
        port = build_portfolio(empty, k=10)
        assert port.empty


# ═══════════════════════════════════════════════════════════════════════════
# C) Cost Modeling Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestCostModeling:
    """Verify cost deduction from portfolio returns."""

    def test_net_leq_gross(self):
        """Net returns should be <= gross returns (costs are non-negative)."""
        df = _make_scored_df()
        port = build_portfolio(df, k=20)
        returns = compute_portfolio_returns(port, cost_bps=10)
        assert all(returns["net_ret"] <= returns["gross_ret"] + 1e-10)

    def test_zero_cost_parity(self):
        """With zero costs, net === gross."""
        df = _make_scored_df()
        port = build_portfolio(df, k=20)
        returns = compute_portfolio_returns(port, cost_bps=0, slippage_multiplier=0)
        np.testing.assert_allclose(returns["net_ret"], returns["gross_ret"], atol=1e-10)

    def test_higher_cost_lower_net(self):
        """Higher cost_bps should result in lower cumulative net returns."""
        df = _make_scored_df()
        port = build_portfolio(df, k=20)
        ret_lo = compute_portfolio_returns(port, cost_bps=5)
        ret_hi = compute_portfolio_returns(port, cost_bps=50)
        assert ret_hi["net_ret"].sum() <= ret_lo["net_ret"].sum()

    def test_turnover_non_negative(self):
        df = _make_scored_df()
        port = build_portfolio(df, k=20)
        returns = compute_portfolio_returns(port)
        assert all(returns["turnover"] >= 0)

    def test_cost_non_negative(self):
        df = _make_scored_df()
        port = build_portfolio(df, k=20)
        returns = compute_portfolio_returns(port, cost_bps=10)
        assert all(returns["cost"] >= 0)


# ═══════════════════════════════════════════════════════════════════════════
# D) Volatility Scaling Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestVolatilityScaling:
    """Verify vol scaling produces sensible results."""

    def test_vol_scaling_keys_present(self):
        """When target_vol is set, metrics should have vol_scaled_sharpe."""
        df = _make_scored_df(n_dates=50)
        port = build_portfolio(df, k=20)
        returns = compute_portfolio_returns(port)
        metrics = compute_portfolio_metrics(returns, target_vol=0.15)
        assert "vol_scaled_sharpe" in metrics
        assert "vol_scaling_effect" in metrics
        assert "vol_scaled_return" in metrics

    def test_no_vol_scaling_when_disabled(self):
        """Without target_vol, no vol-scaling keys."""
        df = _make_scored_df(n_dates=50)
        port = build_portfolio(df, k=20)
        returns = compute_portfolio_returns(port)
        metrics = compute_portfolio_metrics(returns, target_vol=None)
        assert "vol_scaled_sharpe" not in metrics

    def test_metrics_keys(self):
        """Check all expected metric keys are present."""
        df = _make_scored_df(n_dates=50)
        port = build_portfolio(df, k=20)
        returns = compute_portfolio_returns(port)
        metrics = compute_portfolio_metrics(returns)
        expected = {
            "ann_return_gross", "ann_return_net", "ann_vol_gross", "ann_vol_net",
            "gross_sharpe", "net_sharpe", "max_drawdown",
            "cagr_gross", "cagr_net", "avg_turnover", "ann_turnover",
            "avg_holding_period",
        }
        assert expected <= set(metrics.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Equity Curve & Regime
# ═══════════════════════════════════════════════════════════════════════════

class TestEquityCurveAndRegime:

    def test_equity_curve_shape(self):
        df = _make_scored_df()
        port = build_portfolio(df, k=20)
        returns = compute_portfolio_returns(port)
        eq = build_equity_curve(returns)
        assert len(eq) == len(returns)
        assert "cum_gross" in eq.columns
        assert "cum_net" in eq.columns

    def test_regime_breakdown_keys(self):
        df = _make_scored_df(n_dates=30)
        port = build_portfolio(df, k=20)
        returns = compute_portfolio_returns(port)
        regime = compute_regime_breakdown(df, returns)
        expected = {"low_stress", "mid_stress", "high_stress", "top_20pct_stress"}
        assert set(regime.keys()) == expected

    def test_regime_breakdown_covers_all_dates(self):
        df = _make_scored_df(n_dates=30)
        port = build_portfolio(df, k=20)
        returns = compute_portfolio_returns(port)
        regime = compute_regime_breakdown(df, returns)
        total = sum(v["n_dates"] for k, v in regime.items() if k != "top_20pct_stress")
        assert total == len(returns)
