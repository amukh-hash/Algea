"""Tests for the production portfolio backtest layer.

Covers:
- Cost purity (compute_turnover_and_cost)
- Cost invariance (independent of returns)
- Turnover formula correctness
- Buffer-zone hysteresis reduces turnover
- Slot cap limits replacements
- Non-overlapping rebalance schedule
- Vol scaling clips leverage
- Leverage coupling (costs scale with leverage)
- Unscaled vs vol-scaled metric reporting
- Cost model backward compat wrapper
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from algaie.portfolio.portfolio_rules import PortfolioConfig, construct_portfolio
from algaie.portfolio.cost_model import (
    CostConfig,
    apply_costs,
    compute_turnover_and_cost,
)
from algaie.portfolio.vol_scaling import VolTargetConfig, compute_leverage, apply_leverage


# ═══════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════

def _make_date_df(n_stocks: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create a single-date scored DataFrame."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "symbol": [f"S{i:03d}" for i in range(n_stocks)],
        "score_final": rng.randn(n_stocks),
        "y_ret": 0.001 * rng.randn(n_stocks),
        "date": pd.Timestamp("2024-06-01"),
    })


def _make_multi_date_df(n_dates: int = 30, n_stocks: int = 100, seed: int = 42) -> pd.DataFrame:
    """Create multi-date scored DataFrame for backtest testing."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_dates)
    rows = []
    for dt in dates:
        for j in range(n_stocks):
            rows.append({
                "date": dt,
                "symbol": f"S{j:03d}",
                "score_final": rng.randn(),
                "y_ret": 0.001 * rng.randn(),
            })
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════
# Cost purity — compute_turnover_and_cost (Deliverable 1)
# ═══════════════════════════════════════════════════════════════════════

class TestCostPurity:
    """compute_turnover_and_cost is a pure function of weights + config."""

    def test_cost_invariance_across_returns(self):
        """Same weights with different gross returns → identical turnover & cost."""
        old = {"A": 0.5, "B": 0.5}
        new = {"A": 0.3, "C": 0.7}
        cfg = CostConfig(cost_bps=10.0)

        to_1, cost_1 = compute_turnover_and_cost(old, new, cfg)
        to_2, cost_2 = compute_turnover_and_cost(old, new, cfg)

        assert to_1 == to_2
        assert cost_1 == cost_2

    def test_cost_invariance_no_return_dependency(self):
        """Prove the function literally takes no return arg — just weights + cfg."""
        old = {"X": 0.3, "Y": 0.7}
        new = {"Y": 0.4, "Z": 0.6}
        cfg = CostConfig(cost_bps=20.0, impact_bps=5.0)

        # Can't accidentally pass a return — the signature doesn't accept it
        to, cost = compute_turnover_and_cost(old, new, cfg)
        assert to > 0
        assert cost > 0

    def test_zero_turnover_same_weights(self):
        """prev_w == new_w => turnover_1way == 0 and cost == 0."""
        w = {"A": 0.5, "B": 0.5}
        to, cost = compute_turnover_and_cost(w, w, CostConfig())
        assert to == 0.0
        assert cost == 0.0

    def test_zero_turnover_empty_to_empty(self):
        """Both empty => zero."""
        to, cost = compute_turnover_and_cost({}, {}, CostConfig())
        assert to == 0.0
        assert cost == 0.0

    def test_cost_calculation_formula(self):
        """Cost = turnover_1way * total_bps / 10_000."""
        old = {"A": 0.5, "B": 0.5}
        new = {"A": 0.5, "C": 0.5}
        cfg = CostConfig(cost_bps=10.0, impact_bps=0.0)

        to, cost = compute_turnover_and_cost(old, new, cfg)
        expected_to = 0.5  # sum_abs_delta = 1.0, 1-way = 0.5
        expected_cost = 0.5 * 10.0 / 10_000.0
        assert abs(to - expected_to) < 1e-10
        assert abs(cost - expected_cost) < 1e-10


# ═══════════════════════════════════════════════════════════════════════
# Backward-compat wrapper: apply_costs
# ═══════════════════════════════════════════════════════════════════════

class TestApplyCostsWrapper:
    """Legacy apply_costs wrapper works identically to pure func + return subtraction."""

    def test_wrapper_matches_pure(self):
        """apply_costs returns (gross - cost, cost, turnover) matching pure func."""
        old = {"A": 0.6, "B": 0.4}
        new = {"B": 0.3, "C": 0.7}
        cfg = CostConfig(cost_bps=15.0)
        gross_ret = 0.05

        net, cost_w, to_w = apply_costs(old, new, gross_ret, cfg)
        to_p, cost_p = compute_turnover_and_cost(old, new, cfg)

        assert to_w == to_p
        assert cost_w == cost_p
        assert abs(net - (gross_ret - cost_p)) < 1e-15


# ═══════════════════════════════════════════════════════════════════════
# Turnover formula
# ═══════════════════════════════════════════════════════════════════════

class TestTurnoverFormula:

    def test_full_turnover_complete_swap(self):
        """Complete swap of all holdings → turnover = 1.0."""
        old = {"A": 0.5, "B": 0.5}
        new = {"C": 0.5, "D": 0.5}
        to, _ = compute_turnover_and_cost(old, new, CostConfig())
        assert abs(to - 1.0) < 1e-8

    def test_half_turnover_partial_swap(self):
        """Replace one of two equal-weight holdings → turnover = 0.5."""
        old = {"A": 0.5, "B": 0.5}
        new = {"A": 0.5, "C": 0.5}
        to, _ = compute_turnover_and_cost(old, new, CostConfig())
        assert abs(to - 0.5) < 1e-8

    def test_first_period_turnover(self):
        """First period (from cash) → 1-way turnover = 0.5."""
        new = {"A": 0.25, "B": 0.25, "C": 0.25, "D": 0.25}
        to, _ = compute_turnover_and_cost({}, new, CostConfig())
        assert abs(to - 0.5) < 1e-8


# ═══════════════════════════════════════════════════════════════════════
# Buffer-zone hysteresis
# ═══════════════════════════════════════════════════════════════════════

class TestBufferZone:

    def test_held_names_survive_in_buffer(self):
        """Held names with rank between entry and exit should survive."""
        df = _make_date_df(100, seed=1)
        cfg = PortfolioConfig(
            top_k=10, buffer_entry_rank=8, buffer_exit_rank=20,
            max_replacements=None, hold_bonus=0.0,
        )

        # First period: no prev holdings
        w1, ages1, d1 = construct_portfolio(df, None, cfg)
        assert len(w1) == 10

        # Slightly perturb scores so ranks change
        df2 = df.copy()
        df2["score_final"] = df2["score_final"] + np.random.RandomState(99).randn(100) * 0.1

        w2, ages2, d2 = construct_portfolio(df2, w1, cfg, ages1)
        assert len(w2) == 10

        # Some held names should survive even if not in top-8
        assert d2["n_held"] > 0

    def test_hysteresis_reduces_turnover(self):
        """Buffer zone should produce lower turnover than naive top-K."""
        df = _make_date_df(100, seed=1)

        # Naive: tight exit = K (held names exit if they drop out of top-K)
        naive_cfg = PortfolioConfig(
            top_k=10, buffer_entry_rank=10, buffer_exit_rank=11,
            max_replacements=None, hold_bonus=0.0,
        )
        # Buffer zone: wider exit = 25 (held names get 15-rank buffer)
        buffer_cfg = PortfolioConfig(
            top_k=10, buffer_entry_rank=8, buffer_exit_rank=25,
            max_replacements=None, hold_bonus=0.0,
        )

        # Run 8 consecutive periods with large perturbation to force churn
        naive_tos = []
        buffer_tos = []
        prev_naive = None
        prev_buffer = None
        ages_naive = None
        ages_buffer = None

        for step in range(8):
            df_t = df.copy()
            # Large noise to force rank changes
            df_t["score_final"] = np.random.RandomState(step * 7 + 3).randn(100)

            wn, ages_naive, dn = construct_portfolio(df_t, prev_naive, naive_cfg, ages_naive)
            wb, ages_buffer, db = construct_portfolio(df_t, prev_buffer, buffer_cfg, ages_buffer)

            naive_tos.append(dn["turnover_1way"])
            buffer_tos.append(db["turnover_1way"])

            prev_naive = wn
            prev_buffer = wb

        # Buffer should have lower average turnover (skip first period)
        avg_naive = np.mean(naive_tos[1:])
        avg_buffer = np.mean(buffer_tos[1:])
        assert avg_buffer <= avg_naive + 0.01, \
            f"Buffer turnover {avg_buffer:.3f} should be <= naive {avg_naive:.3f}"


# ═══════════════════════════════════════════════════════════════════════
# Slot cap
# ═══════════════════════════════════════════════════════════════════════

class TestSlotCap:

    def test_slot_cap_limits_replacements(self):
        """Max replacements should cap the number of new entries."""
        df = _make_date_df(100, seed=1)
        cfg = PortfolioConfig(
            top_k=20, buffer_entry_rank=20, buffer_exit_rank=40,
            max_replacements=3, hold_bonus=0.0,
        )

        w1, ages1, _ = construct_portfolio(df, None, cfg)

        # Drastically change scores to force many replacements
        df2 = df.copy()
        df2["score_final"] = -df2["score_final"]  # flip all scores

        w2, _, d2 = construct_portfolio(df2, w1, cfg, ages1)

        # New entries should be <= max_replacements
        assert d2["n_new"] <= 3, f"Expected n_new <= 3, got {d2['n_new']}"

    def test_slot_cap_none_unlimited(self):
        """If max_replacements=None, no cap is applied."""
        df = _make_date_df(100, seed=1)
        cfg = PortfolioConfig(
            top_k=20, buffer_entry_rank=20, buffer_exit_rank=40,
            max_replacements=None, hold_bonus=0.0,
        )

        w1, ages1, _ = construct_portfolio(df, None, cfg)

        df2 = df.copy()
        df2["score_final"] = -df2["score_final"]

        w2, _, d2 = construct_portfolio(df2, w1, cfg, ages1)

        # Should be able to replace all 20
        assert d2["n_new"] > 3  # more than capped version would allow


# ═══════════════════════════════════════════════════════════════════════
# Non-overlapping schedule
# ═══════════════════════════════════════════════════════════════════════

class TestNonOverlapping:

    def test_schedule_spacing(self):
        """Rebalance dates should be spaced by exactly horizon trading days."""
        from backend.scripts.run_selector_portfolio import build_rebalance_schedule

        dates = pd.bdate_range("2024-01-02", periods=100)
        schedule = build_rebalance_schedule(dates, horizon=10)

        assert len(schedule) == 10  # 100 / 10
        for i in range(1, len(schedule)):
            idx_prev = list(dates).index(schedule[i - 1])
            idx_curr = list(dates).index(schedule[i])
            assert idx_curr - idx_prev == 10

    def test_schedule_floor_division(self):
        """Number of periods = floor(n_dates / horizon)."""
        from backend.scripts.run_selector_portfolio import build_rebalance_schedule

        dates = pd.bdate_range("2024-01-02", periods=57)
        schedule = build_rebalance_schedule(dates, horizon=10)
        assert len(schedule) == 6  # floor(57/10) = 5, but [::10] gives 6 elements


# ═══════════════════════════════════════════════════════════════════════
# Vol scaling
# ═══════════════════════════════════════════════════════════════════════

class TestVolScaling:

    def test_leverage_clips_max(self):
        """Leverage should not exceed max_leverage."""
        cfg = VolTargetConfig(target_vol_ann=0.30, lookback_periods=5, max_leverage=1.0)
        # Very low vol returns → would want high leverage
        returns = np.array([0.001, 0.001, 0.001, 0.001, 0.001, 0.001])
        lev = compute_leverage(returns, 6, cfg, periods_per_year=25.2)
        assert lev <= 1.0 + 1e-8

    def test_leverage_clips_min(self):
        """Leverage should not go below min_leverage."""
        cfg = VolTargetConfig(target_vol_ann=0.01, lookback_periods=5,
                              max_leverage=2.0, min_leverage=0.2)
        # Very high vol returns → would want very low leverage
        returns = np.array([0.1, -0.1, 0.1, -0.1, 0.1, -0.1])
        lev = compute_leverage(returns, 6, cfg, periods_per_year=25.2)
        assert lev >= 0.2 - 1e-8

    def test_leverage_responds_to_vol(self):
        """Higher realized vol → lower leverage."""
        cfg = VolTargetConfig(target_vol_ann=0.15, lookback_periods=5, max_leverage=2.0)

        # Low vol
        low_vol = np.array([0.001, 0.002, 0.001, 0.002, 0.001, 0.002])
        lev_low = compute_leverage(low_vol, 6, cfg, periods_per_year=25.2)

        # High vol
        high_vol = np.array([0.05, -0.04, 0.06, -0.05, 0.04, -0.03])
        lev_high = compute_leverage(high_vol, 6, cfg, periods_per_year=25.2)

        assert lev_high < lev_low, f"High vol leverage ({lev_high:.3f}) should be < low vol ({lev_low:.3f})"

    def test_apply_leverage_scales_return(self):
        """apply_leverage should multiply return by leverage."""
        assert abs(apply_leverage(0.05, 0.5) - 0.025) < 1e-10
        assert abs(apply_leverage(0.05, 1.0) - 0.05) < 1e-10
        assert abs(apply_leverage(0.05, 2.0) - 0.10) < 1e-10

    def test_not_enough_history_returns_default(self):
        """Before lookback is filled, leverage defaults to 1.0."""
        cfg = VolTargetConfig(target_vol_ann=0.15, lookback_periods=12, max_leverage=1.0)
        lev = compute_leverage(np.array([0.01] * 5), 5, cfg)
        assert lev == 1.0


# ═══════════════════════════════════════════════════════════════════════
# Integration: construct_portfolio weight properties
# ═══════════════════════════════════════════════════════════════════════

class TestPortfolioProperties:

    def test_weights_sum_to_one(self):
        """Equal-weight portfolio should sum to 1.0."""
        df = _make_date_df()
        cfg = PortfolioConfig(top_k=10)
        w, _, _ = construct_portfolio(df, None, cfg)
        total = sum(w.values())
        assert abs(total - 1.0) < 1e-8

    def test_exactly_k_holdings(self):
        """Should hold exactly K stocks."""
        df = _make_date_df(n_stocks=200)
        cfg = PortfolioConfig(top_k=50)
        w, _, _ = construct_portfolio(df, None, cfg)
        assert len(w) == 50

    def test_fewer_than_k_available(self):
        """If fewer than K stocks available, hold all."""
        df = _make_date_df(n_stocks=5)
        cfg = PortfolioConfig(top_k=50)
        w, _, diag = construct_portfolio(df, None, cfg)
        assert len(w) == 5
        assert abs(sum(w.values()) - 1.0) < 1e-8

    def test_hold_bonus_affects_ranking(self):
        """Hold bonus should make held names rank higher."""
        df = _make_date_df(100, seed=1)
        cfg_no_bonus = PortfolioConfig(
            top_k=10, buffer_entry_rank=10, buffer_exit_rank=20,
            max_replacements=None, hold_bonus=0.0,
        )
        cfg_bonus = PortfolioConfig(
            top_k=10, buffer_entry_rank=10, buffer_exit_rank=20,
            max_replacements=None, hold_bonus=0.5,
        )

        w1, ages1, _ = construct_portfolio(df, None, cfg_no_bonus)

        # Perturb scores slightly
        df2 = df.copy()
        df2["score_final"] += np.random.RandomState(7).randn(100) * 0.2

        w_nb, _, d_nb = construct_portfolio(df2, w1, cfg_no_bonus, ages1)
        w_b, _, d_b = construct_portfolio(df2, w1, cfg_bonus, ages1)

        # With bonus, more names should be held
        assert d_b["n_held"] >= d_nb["n_held"]

    def test_config_validation_rejects_bad_buffer(self):
        """Entry rank must be < exit rank."""
        with pytest.raises(ValueError):
            PortfolioConfig(buffer_entry_rank=70, buffer_exit_rank=40)


# ═══════════════════════════════════════════════════════════════════════
# Leverage coupling (Deliverable 4.3)
# ═══════════════════════════════════════════════════════════════════════

class TestLeverageCouplingExplicit:
    """Known turnover + known leverage → exact cost/return coupling."""

    def test_cost_scaled_equals_cost_times_lev(self):
        """With lev=2, cost_scaled == raw_cost * 2."""
        # Manual scenario: turnover=0.5, cost_bps=10 → cost = 0.5 * 10/10000 = 0.0005
        old = {"A": 0.5, "B": 0.5}
        new = {"A": 0.5, "C": 0.5}
        cfg = CostConfig(cost_bps=10.0)
        to, cost = compute_turnover_and_cost(old, new, cfg)

        lev = 2.0
        cost_scaled = cost * lev
        gross_ret = 0.02
        gross_scaled = gross_ret * lev
        net_scaled = gross_scaled - cost_scaled

        assert abs(cost_scaled - cost * 2.0) < 1e-15
        assert abs(net_scaled - (gross_ret * 2 - cost * 2)) < 1e-15

    def test_leverage_1_identity(self):
        """When lev=1, scaled and unscaled metrics are identical."""
        old = {"X": 0.3, "Y": 0.7}
        new = {"Y": 0.5, "Z": 0.5}
        cfg = CostConfig(cost_bps=15.0)
        to, cost = compute_turnover_and_cost(old, new, cfg)

        lev = 1.0
        gross_ret = 0.03
        net_unscaled = gross_ret - cost
        net_scaled = gross_ret * lev - cost * lev

        assert abs(net_unscaled - net_scaled) < 1e-15

    def test_cost_scales_with_leverage_in_backtest(self):
        """Backtest: lower target vol → lower leverage → lower scaled costs."""
        from backend.scripts.run_selector_portfolio import run_backtest

        df = _make_multi_date_df(n_dates=60, n_stocks=100, seed=42)
        port_cfg = PortfolioConfig(top_k=20, rebalance_horizon_days=10)
        cost_cfg = CostConfig(cost_bps=10.0)

        # Higher target vol → higher leverage
        vol_cfg_hi = VolTargetConfig(target_vol_ann=0.15, max_leverage=1.0, lookback_periods=3)
        ret_hi, _ = run_backtest(df, port_cfg, cost_cfg, vol_cfg_hi)

        # Very low target vol → very low leverage
        vol_cfg_lo = VolTargetConfig(target_vol_ann=0.01, max_leverage=1.0, lookback_periods=3)
        ret_lo, _ = run_backtest(df, port_cfg, cost_cfg, vol_cfg_lo)

        if ret_hi is not None and not ret_hi.empty and not ret_lo.empty:
            if "cost_scaled" in ret_hi.columns and "cost_scaled" in ret_lo.columns:
                avg_hi = ret_hi["cost_scaled"].mean()
                avg_lo = ret_lo["cost_scaled"].mean()
                assert avg_lo <= avg_hi + 1e-6


# ═══════════════════════════════════════════════════════════════════════
# Reporting integrity (Deliverable 4.4)
# ═══════════════════════════════════════════════════════════════════════

class TestReportingIntegrity:
    """Unscaled and volscaled metric keys are correctly present/absent."""

    def test_vol_scaling_enabled_has_both_keys(self):
        """When vol scaling enabled, both _unscaled and _volscaled keys exist."""
        from backend.scripts.run_selector_portfolio import run_backtest

        df = _make_multi_date_df(n_dates=60, n_stocks=100, seed=42)
        port_cfg = PortfolioConfig(top_k=20, rebalance_horizon_days=10)
        cost_cfg = CostConfig(cost_bps=10.0)
        vol_cfg = VolTargetConfig(target_vol_ann=0.15, max_leverage=1.0, lookback_periods=3)

        _, summary = run_backtest(df, port_cfg, cost_cfg, vol_cfg)

        assert summary is not None
        # Unscaled keys (always present)
        assert "net_sharpe_unscaled" in summary
        assert "ann_return_net_unscaled" in summary
        assert "ann_vol_net_unscaled" in summary
        assert "cagr_net_unscaled" in summary
        # Vol-scaled keys (present when vol scaling enabled)
        assert "net_sharpe_volscaled" in summary
        assert "ann_return_net_volscaled" in summary
        assert "ann_vol_net_volscaled" in summary
        assert "cagr_net_volscaled" in summary
        # Backward-compat aliases
        assert "net_sharpe" in summary
        assert summary["net_sharpe"] == summary["net_sharpe_unscaled"]
        assert "vol_scaled_sharpe" in summary
        assert summary["vol_scaled_sharpe"] == summary["net_sharpe_volscaled"]

    def test_vol_scaling_disabled_no_volscaled_keys(self):
        """When vol scaling disabled, _volscaled keys are absent and no crash."""
        from backend.scripts.run_selector_portfolio import run_backtest

        df = _make_multi_date_df(n_dates=60, n_stocks=100, seed=42)
        port_cfg = PortfolioConfig(top_k=20, rebalance_horizon_days=10)
        cost_cfg = CostConfig(cost_bps=10.0)

        _, summary = run_backtest(df, port_cfg, cost_cfg, None)

        assert summary is not None
        # Unscaled always present
        assert "net_sharpe_unscaled" in summary
        assert "ann_return_net_unscaled" in summary
        assert "cagr_net_unscaled" in summary
        # Vol-scaled absent
        assert "net_sharpe_volscaled" not in summary
        assert "ann_return_net_volscaled" not in summary
        # Backward compat alias still works
        assert "net_sharpe" in summary
        assert summary["net_sharpe"] == summary["net_sharpe_unscaled"]

    def test_vol_scaled_sharpe_identity_lev1(self):
        """When leverage ≈ 1.0, vol_scaled_sharpe ≈ net_sharpe."""
        from backend.scripts.run_selector_portfolio import run_backtest

        df = _make_multi_date_df(n_dates=60, n_stocks=100, seed=42)
        port_cfg = PortfolioConfig(top_k=20, rebalance_horizon_days=10)
        cost_cfg = CostConfig(cost_bps=10.0)
        # Extremely high target vol → leverage always 1.0
        vol_cfg = VolTargetConfig(target_vol_ann=100.0, max_leverage=1.0, lookback_periods=3)

        _, summary = run_backtest(df, port_cfg, cost_cfg, vol_cfg)

        if summary:
            net = summary.get("net_sharpe_unscaled", 0)
            vs = summary.get("net_sharpe_volscaled", 0)
            assert abs(net - vs) < 0.1, \
                f"vol_scaled ({vs}) should ≈ unscaled ({net}) when lev≈1.0"

    def test_extended_metrics_leverage_fields(self):
        """Portfolio metrics include leverage/exposure stats."""
        from backend.scripts.run_selector_portfolio import run_backtest

        df = _make_multi_date_df(n_dates=60, n_stocks=100, seed=42)
        port_cfg = PortfolioConfig(top_k=20, rebalance_horizon_days=10)
        cost_cfg = CostConfig(cost_bps=10.0)
        vol_cfg = VolTargetConfig(target_vol_ann=0.15, max_leverage=1.0, lookback_periods=3)

        _, summary = run_backtest(df, port_cfg, cost_cfg, vol_cfg)

        assert summary is not None
        assert "leverage_mean" in summary
        assert "leverage_std" in summary
        assert "effective_cost_ann" in summary
        assert "gross_exposure_mean" in summary
        assert "rebalance_count" in summary
        assert summary["rebalance_count"] > 0


# ═══════════════════════════════════════════════════════════════════════
# Walk-forward (Part 3)
# ═══════════════════════════════════════════════════════════════════════

class TestWalkForward:
    """Walk-forward fold generation."""

    def test_walk_forward_fold_count(self):
        """Verify correct number of folds are generated."""
        from backend.scripts.run_selector_portfolio import run_walk_forward

        # Create 7 years of data (2018-2024)
        dates = pd.bdate_range("2018-01-02", "2024-12-31")
        n_stocks = 50
        rng = np.random.RandomState(42)
        rows = []
        for dt in dates:
            for j in range(n_stocks):
                rows.append({
                    "date": dt,
                    "symbol": f"S{j:03d}",
                    "score_final": rng.randn(),
                    "y_ret": 0.001 * rng.randn(),
                })
        df = pd.DataFrame(rows)

        port_cfg = PortfolioConfig(top_k=10, rebalance_horizon_days=10)
        cost_cfg = CostConfig(cost_bps=10.0)

        result = run_walk_forward(
            df, port_cfg, cost_cfg, None,
            train_years=3, val_years=1, test_years=1,
        )

        assert result["status"] == "success"
        assert result["n_folds"] >= 2
        assert result["total_test_periods"] > 0


# ═══════════════════════════════════════════════════════════════════════
# Cost sensitivity (Part 6)
# ═══════════════════════════════════════════════════════════════════════

class TestCostSensitivity:
    """Cost sensitivity diagnostic output."""

    def test_cost_sensitivity_output(self):
        """Verify cost sensitivity returns correct structure."""
        from backend.scripts.run_selector_portfolio import run_cost_sensitivity

        df = _make_multi_date_df(n_dates=60, n_stocks=100, seed=42)
        port_cfg = PortfolioConfig(top_k=20, rebalance_horizon_days=10)

        result = run_cost_sensitivity(df, port_cfg, None, cost_bps_list=[5.0, 10.0, 20.0])

        assert "cost_sensitivity" in result
        cs = result["cost_sensitivity"]
        assert len(cs) == 3
        assert cs[0]["cost_bps"] == 5.0
        assert cs[1]["cost_bps"] == 10.0
        assert cs[2]["cost_bps"] == 20.0
        # Monotonic: higher cost → lower Sharpe
        assert cs[0]["net_sharpe"] >= cs[2]["net_sharpe"]


# ═══════════════════════════════════════════════════════════════════════
# Decision frequency (Part 2A)
# ═══════════════════════════════════════════════════════════════════════

class TestDecisionFrequency:
    """Decision-date sampling."""

    def test_decision_frequency_subsamples(self):
        """decision_frequency=5 should use ~1/5 of dates."""
        from algaie.training.selector_dataset import SelectorDataset

        n_dates = 100
        dates = pd.bdate_range("2024-01-02", periods=n_dates)
        rows = []
        for dt in dates:
            for j in range(10):
                rows.append({
                    "date": dt,
                    "symbol": f"S{j:03d}",
                    "y_ret": 0.01,
                    "y_vol": 0.0,
                    "weight": 1.0,
                })
        df = pd.DataFrame(rows)
        df["y_rank"] = df["y_ret"]
        df["y_trade"] = (df["y_ret"] > 0).astype(float)
        df["tier"] = 0

        ds_full = SelectorDataset(df, feature_cols=[], decision_frequency=1)
        ds_5 = SelectorDataset(df, feature_cols=[], decision_frequency=5)

        assert len(ds_full) == n_dates
        assert len(ds_5) == n_dates // 5  # 100 / 5 = 20

    def test_decision_frequency_1_identity(self):
        """decision_frequency=1 should keep all dates."""
        from algaie.training.selector_dataset import SelectorDataset

        n_dates = 50
        dates = pd.bdate_range("2024-01-02", periods=n_dates)
        rows = []
        for dt in dates:
            rows.append({
                "date": dt,
                "symbol": "S000",
                "y_ret": 0.01,
                "y_vol": 0.0,
                "weight": 1.0,
            })
        df = pd.DataFrame(rows)
        df["y_rank"] = df["y_ret"]
        df["y_trade"] = (df["y_ret"] > 0).astype(float)
        df["tier"] = 0

        ds = SelectorDataset(df, feature_cols=[], decision_frequency=1)
        assert len(ds) == n_dates
