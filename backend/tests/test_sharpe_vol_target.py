"""
Tests for Vol Target Scaling Study — one class per phase + end-to-end.
"""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.analysis.sharpe_vol_target import (
    compute_baseline_metrics,
    compute_scaling_factor,
    scale_returns,
    compute_scaled_metrics,
    cost_adjusted_sharpe,
    compute_cost_adjusted_table,
    nonlinear_risk_adjustment,
    build_comparison_table,
    run_vol_target_study,
)
from backend.analysis.sharpe_report import compute_sharpe, _ann_vol
from backend.analysis.sharpe_validator import validate_and_load


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def _make_df(N=300, seed=42):
    """Synthetic data through validate_and_load."""
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-01", periods=N)
    data = pd.DataFrame({
        "date": dates,
        "core_pnl": np.random.normal(500, 2000, N),
        "vrp_pnl": np.random.normal(200, 1000, N),
        "w_vrp": np.clip(np.random.uniform(0.05, 0.25, N), 0, 0.25),
        "regime": np.random.choice(
            ["normal_carry", "caution", "crash_risk"], N, p=[0.6, 0.3, 0.1]
        ),
    })
    return validate_and_load(data)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Baseline Metrics
# ═══════════════════════════════════════════════════════════════════════════

class TestBaselineMetrics:

    def test_structure(self):
        """Has all expected keys."""
        df = _make_df()
        bl = compute_baseline_metrics(df["port_ret"])
        expected = {"ann_return", "ann_vol", "sharpe", "es_95", "max_drawdown", "n_obs"}
        assert expected == set(bl.keys())

    def test_vol_matches(self):
        """Baseline vol matches _ann_vol computation."""
        df = _make_df()
        bl = compute_baseline_metrics(df["port_ret"])
        expected_vol = float(_ann_vol(df["port_ret"]))
        assert abs(bl["ann_vol"] - expected_vol) < 1e-6

    def test_sharpe_matches(self):
        """Baseline Sharpe matches compute_sharpe."""
        df = _make_df()
        bl = compute_baseline_metrics(df["port_ret"])
        expected_sh = float(compute_sharpe(df["port_ret"]))
        assert abs(bl["sharpe"] - expected_sh) < 1e-4

    def test_es_negative(self):
        """ES95 should be negative for typical returns."""
        df = _make_df()
        bl = compute_baseline_metrics(df["port_ret"])
        assert bl["es_95"] < 0

    def test_max_drawdown_negative(self):
        """Max drawdown should be <= 0."""
        df = _make_df()
        bl = compute_baseline_metrics(df["port_ret"])
        assert bl["max_drawdown"] <= 0


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Scaling
# ═══════════════════════════════════════════════════════════════════════════

class TestScaling:

    def test_scaling_factor_correct(self):
        """scale = target / baseline."""
        assert abs(compute_scaling_factor(0.025, 0.10) - 4.0) < 1e-10

    def test_scaling_factor_zero_vol(self):
        """Zero baseline vol → scale = 0."""
        assert compute_scaling_factor(0.0, 0.10) == 0.0

    def test_scale_returns_linear(self):
        """Scaled returns = original * scale."""
        r = pd.Series([0.01, -0.02, 0.005])
        scaled = scale_returns(r, 4.0)
        np.testing.assert_allclose(scaled.values, [0.04, -0.08, 0.02])

    def test_scaled_vol_matches_target(self):
        """After scaling, annualized vol should be close to target."""
        df = _make_df()
        returns = df["port_ret"]
        baseline_vol = float(_ann_vol(returns))
        target = 0.10
        scale = compute_scaling_factor(baseline_vol, target)
        scaled_ret = scale_returns(returns, scale)
        actual_vol = float(_ann_vol(scaled_ret))
        assert abs(actual_vol - target) / target < 0.01, \
            f"Scaled vol {actual_vol} should be ~{target}"

    def test_sharpe_preserved_under_scaling(self):
        """Sharpe ratio should be approximately invariant under linear scaling."""
        df = _make_df()
        returns = df["port_ret"]
        baseline_sharpe = float(compute_sharpe(returns))
        baseline_vol = float(_ann_vol(returns))

        for target in [0.05, 0.08, 0.10, 0.12]:
            metrics = compute_scaled_metrics(returns, target, baseline_vol)
            pct_diff = abs(metrics["sharpe"] - baseline_sharpe) / max(abs(baseline_sharpe), 1e-8) * 100
            assert pct_diff < 2.0, \
                f"Sharpe changed by {pct_diff:.1f}% at {target*100}% vol"

    def test_scaled_metrics_structure(self):
        """compute_scaled_metrics returns all expected keys."""
        df = _make_df()
        returns = df["port_ret"]
        baseline_vol = float(_ann_vol(returns))
        m = compute_scaled_metrics(returns, 0.10, baseline_vol)
        expected = {"target_vol", "scale_factor", "ann_return", "ann_vol",
                    "sharpe", "es_95", "max_drawdown"}
        assert expected == set(m.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Cost Adjustment
# ═══════════════════════════════════════════════════════════════════════════

class TestCostAdjustment:

    def test_cost_reduces_sharpe(self):
        """Adding cost should reduce Sharpe."""
        df = _make_df()
        returns = df["port_ret"]
        turnover = df["w_vrp"].diff().abs()
        turnover.iloc[0] = abs(df["w_vrp"].iloc[0])
        baseline_vol = float(_ann_vol(returns))
        scale = compute_scaling_factor(baseline_vol, 0.10)

        sh_0 = float(compute_sharpe(returns * scale))
        sh_10 = cost_adjusted_sharpe(returns, scale, turnover, 10.0)
        assert sh_10 < sh_0

    def test_higher_cost_lower_sharpe(self):
        """More cost → worse Sharpe."""
        df = _make_df()
        returns = df["port_ret"]
        turnover = df["w_vrp"].diff().abs()
        turnover.iloc[0] = abs(df["w_vrp"].iloc[0])
        baseline_vol = float(_ann_vol(returns))
        scale = compute_scaling_factor(baseline_vol, 0.10)

        sh_5 = cost_adjusted_sharpe(returns, scale, turnover, 5.0)
        sh_10 = cost_adjusted_sharpe(returns, scale, turnover, 10.0)
        sh_20 = cost_adjusted_sharpe(returns, scale, turnover, 20.0)
        assert sh_5 >= sh_10 >= sh_20

    def test_cost_table_structure(self):
        """Cost table has all vol targets and cost scenarios."""
        df = _make_df()
        returns = df["port_ret"]
        turnover = df["w_vrp"].diff().abs()
        turnover.iloc[0] = abs(df["w_vrp"].iloc[0])
        baseline_vol = float(_ann_vol(returns))

        table = compute_cost_adjusted_table(
            returns, turnover, [0.05, 0.10], baseline_vol, [5.0, 10.0]
        )
        assert "5%" in table
        assert "10%" in table
        assert "+5bps" in table["5%"]
        assert "+10bps" in table["10%"]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — Nonlinear Risk
# ═══════════════════════════════════════════════════════════════════════════

class TestNonlinearRisk:

    def test_es_scales_linearly(self):
        """ES_scaled = ES * scale."""
        result = nonlinear_risk_adjustment(-0.01, -0.05, 4.0)
        assert abs(result["es_95_scaled"] - (-0.04)) < 1e-10
        assert abs(result["max_drawdown_scaled"] - (-0.20)) < 1e-10

    def test_dd_flag_at_threshold(self):
        """MaxDD > 20% → flag."""
        result = nonlinear_risk_adjustment(-0.01, -0.06, 4.0)
        assert result["max_dd_exceeds_20pct"] is True  # 6% * 4 = 24%

    def test_dd_no_flag_below(self):
        """MaxDD < 20% → no flag."""
        result = nonlinear_risk_adjustment(-0.01, -0.04, 4.0)
        assert result["max_dd_exceeds_20pct"] is False  # 4% * 4 = 16%


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5 — Comparison Table & End-to-End
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEnd:

    def test_report_structure(self):
        """Full report has all expected keys."""
        df = _make_df()
        report = run_vol_target_study("e2e", df)
        expected = {"run_id", "sleeve", "baseline", "scaled_scenarios",
                    "cost_adjusted_sharpe", "risk_adjustments",
                    "comparison_table", "verdict"}
        assert expected == set(report.keys())

    def test_comparison_table_rows(self):
        """Table has 1 baseline + 4 target rows = 5 total."""
        df = _make_df()
        report = run_vol_target_study("tbl", df)
        assert len(report["comparison_table"]) == 5

    def test_verdict_has_classification(self):
        """Verdict has classification and flags."""
        df = _make_df()
        report = run_vol_target_study("v", df)
        assert "classification" in report["verdict"]
        assert "flags" in report["verdict"]

    def test_deterministic(self):
        """Same seed → same results."""
        df1 = _make_df(seed=77)
        df2 = _make_df(seed=77)
        r1 = run_vol_target_study("d1", df1)
        r2 = run_vol_target_study("d2", df2)
        assert r1["baseline"]["sharpe"] == r2["baseline"]["sharpe"]
        assert r1["verdict"]["classification"] == r2["verdict"]["classification"]

    def test_scaled_10_vol_close_to_target(self):
        """10% scenario should have vol ≈ 10%."""
        df = _make_df()
        report = run_vol_target_study("vol", df)
        scaled_10 = report["scaled_scenarios"]["10%"]
        assert abs(scaled_10["ann_vol"] - 0.10) / 0.10 < 0.02

    def test_sharpe_invariance_in_table(self):
        """All rows in comparison table should show similar Sharpe (pre-cost)."""
        df = _make_df()
        report = run_vol_target_study("inv", df)
        sharpes = [r["sharpe"] for r in report["comparison_table"]]
        baseline_sh = sharpes[0]
        for sh in sharpes[1:]:
            pct_diff = abs(sh - baseline_sh) / max(abs(baseline_sh), 1e-8) * 100
            assert pct_diff < 2.0, \
                f"Sharpe should be invariant under scaling, but got {pct_diff:.1f}% diff"
