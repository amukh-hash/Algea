"""
Tests for Sharpe Integrity Audit — one class per phase.
"""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.analysis.sharpe_audit import (
    compute_autocorrelations,
    ljung_box_pvalues,
    newey_west_vol,
    newey_west_sharpe,
    audit_autocorrelation,
    audit_capital_utilization,
    compute_turnover,
    cost_sensitivity_analysis,
    block_bootstrap_sharpe,
    audit_bootstrap,
    audit_rolling_stability,
    audit_pnl_concentration,
    audit_oos_segmentation,
    classify_sharpe_integrity,
    run_sharpe_audit,
)
from backend.analysis.sharpe_report import compute_sharpe
from backend.analysis.sharpe_validator import validate_and_load


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_df(N=300, seed=42):
    """Create synthetic data and run through validate_and_load."""
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


def _make_autocorrelated_returns(N=500, rho=0.5, seed=42):
    """Generate AR(1) returns with known autocorrelation."""
    np.random.seed(seed)
    eps = np.random.normal(0.0005, 0.01, N)
    returns = np.empty(N)
    returns[0] = eps[0]
    for i in range(1, N):
        returns[i] = rho * returns[i - 1] + eps[i]
    return pd.Series(returns)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Autocorrelation
# ═══════════════════════════════════════════════════════════════════════════

class TestAutocorrelation:

    def test_iid_returns_no_autocorrelation(self):
        """IID normal returns → autocorrelations near zero."""
        np.random.seed(42)
        r = pd.Series(np.random.normal(0, 0.01, 500))
        acfs = compute_autocorrelations(r)
        for ac in acfs:
            assert abs(ac) < 0.15, f"IID returns should have ~0 autocorrelation, got {ac}"

    def test_ar1_detects_autocorrelation(self):
        """AR(1) with rho=0.5 → lag-1 autocorrelation near 0.5."""
        r = _make_autocorrelated_returns(rho=0.5)
        acfs = compute_autocorrelations(r)
        assert acfs[0] > 0.3, f"Expected lag-1 AC > 0.3 for AR(1), got {acfs[0]}"

    def test_ljung_box_iid(self):
        """IID returns → Ljung-Box p-values > 0.05."""
        np.random.seed(42)
        r = pd.Series(np.random.normal(0, 0.01, 500))
        pvals = ljung_box_pvalues(r)
        # At least 3 out of 5 should be > 0.05 for IID
        n_ok = sum(1 for p in pvals if p > 0.05)
        assert n_ok >= 3, f"IID should not trigger Ljung-Box, pvals={pvals}"

    def test_ljung_box_ar1(self):
        """AR(1) with strong rho → at least one p-value < 0.05."""
        r = _make_autocorrelated_returns(rho=0.5)
        pvals = ljung_box_pvalues(r)
        assert any(p < 0.05 for p in pvals), f"AR(1) should trigger LB, pvals={pvals}"

    def test_newey_west_vol_iid(self):
        """For IID returns, NW vol should be close to naive vol."""
        np.random.seed(42)
        r = pd.Series(np.random.normal(0, 0.01, 500))
        naive_vol = float(r.std(ddof=1) * np.sqrt(252))
        nw_vol = newey_west_vol(r, max_lag=5)
        pct_diff = abs(nw_vol - naive_vol) / naive_vol * 100
        assert pct_diff < 15, f"NW vol should be close to naive for IID, diff={pct_diff:.1f}%"

    def test_newey_west_vol_ar1_higher(self):
        """For AR(1) with positive rho, NW vol should be higher than naive."""
        r = _make_autocorrelated_returns(rho=0.5, N=1000)
        naive_vol = float(r.std(ddof=1) * np.sqrt(252))
        nw_vol = newey_west_vol(r, max_lag=5)
        assert nw_vol > naive_vol * 1.05, \
            f"NW vol ({nw_vol}) should exceed naive ({naive_vol}) for AR(1)"

    def test_newey_west_sharpe_lower_for_ar1(self):
        """NW-adjusted Sharpe should be lower than naive for AR(1)."""
        r = _make_autocorrelated_returns(rho=0.5, N=1000)
        naive = compute_sharpe(r)
        nw = newey_west_sharpe(r, max_lag=5)
        assert nw < naive, f"NW Sharpe ({nw}) should be < naive ({naive}) for AR(1)"

    def test_audit_autocorrelation_structure(self):
        """audit_autocorrelation returns correct structure."""
        df = _make_df()
        result = audit_autocorrelation(df)
        for label in ["core", "vrp", "combined"]:
            assert label in result
            assert "autocorrelations" in result[label]
            assert "ljung_box_pvalues" in result[label]
            assert "significant_autocorrelation" in result[label]
            assert "naive_sharpe" in result[label]


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Capital Utilization
# ═══════════════════════════════════════════════════════════════════════════

class TestCapitalUtilization:

    def test_low_vol_flags_under_deployment(self):
        """Very low vol → under_deployed_flag."""
        df = _make_df()
        # The synthetic data has ~2.5% vol which is < 5% threshold
        result = audit_capital_utilization(df)
        assert result["under_deployed_flag"] is True

    def test_high_vol_no_flag(self):
        """Higher vol → no flag."""
        np.random.seed(42)
        N = 200
        dates = pd.bdate_range("2023-01-01", periods=N)
        # Use large PnL to push vol > 5%
        data = pd.DataFrame({
            "date": dates,
            "core_pnl": np.random.normal(5000, 50000, N),
            "vrp_pnl": np.random.normal(2000, 20000, N),
            "w_vrp": np.full(N, 0.15),
        })
        df = validate_and_load(data)
        result = audit_capital_utilization(df)
        assert result["under_deployed_flag"] is False

    def test_structure(self):
        """Has all expected keys."""
        df = _make_df()
        result = audit_capital_utilization(df)
        expected = {"mean_w_vrp", "ann_vol_portfolio", "under_deployed_flag",
                    "vol_threshold", "correlation_weight_abs_return", "diagnosis"}
        assert expected == set(result.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Turnover & Cost Sensitivity
# ═══════════════════════════════════════════════════════════════════════════

class TestTurnoverCost:

    def test_constant_weight_zero_turnover(self):
        """Constant weight → zero turnover after initial deployment."""
        df = _make_df()
        df["w_vrp"] = 0.10  # constant
        turnover = compute_turnover(df)
        # Only the initial deployment counts
        expected_daily = 0.10 / len(df)
        assert abs(turnover["mean_daily_turnover"] - expected_daily) < 1e-6

    def test_cost_scenarios_present(self):
        """cost_sensitivity_analysis returns all 3 cost scenarios."""
        df = _make_df()
        result = cost_sensitivity_analysis(df)
        assert "+5bps" in result["cost_scenarios"]
        assert "+10bps" in result["cost_scenarios"]
        assert "+20bps" in result["cost_scenarios"]

    def test_sharpe_decreases_with_cost(self):
        """Higher costs → lower Sharpe."""
        df = _make_df()
        result = cost_sensitivity_analysis(df)
        s_base = result["baseline_sharpe"]
        s_5 = result["cost_scenarios"]["+5bps"]["sharpe"]
        s_10 = result["cost_scenarios"]["+10bps"]["sharpe"]
        s_20 = result["cost_scenarios"]["+20bps"]["sharpe"]
        assert s_5 <= s_base
        assert s_10 <= s_5
        assert s_20 <= s_10


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — Bootstrap
# ═══════════════════════════════════════════════════════════════════════════

class TestBootstrap:

    def test_deterministic(self):
        """Same seed → same bootstrap results."""
        np.random.seed(42)
        r = pd.Series(np.random.normal(0.0003, 0.01, 200))
        b1 = block_bootstrap_sharpe(r, seed=42)
        b2 = block_bootstrap_sharpe(r, seed=42)
        assert b1["mean_sharpe"] == b2["mean_sharpe"]
        assert b1["ci_5"] == b2["ci_5"]

    def test_ci_contains_point(self):
        """Point estimate Sharpe should be within 97.5% CI."""
        df = _make_df()
        r = df["port_ret"]
        point = compute_sharpe(r)
        b = block_bootstrap_sharpe(r, seed=42)
        # 97.5% CI is wide — point should be inside
        assert b["ci_2_5"] <= point <= b["ci_97_5"], \
            f"Point {point} outside [{b['ci_2_5']}, {b['ci_97_5']}]"

    def test_structure(self):
        """Has all expected keys."""
        r = pd.Series(np.random.normal(0, 0.01, 100))
        b = block_bootstrap_sharpe(r)
        expected = {"n_samples", "block_length", "mean_sharpe", "std_sharpe",
                    "ci_5", "ci_95", "ci_2_5", "ci_97_5"}
        assert expected == set(b.keys())


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5 — Rolling Stability
# ═══════════════════════════════════════════════════════════════════════════

class TestRollingStability:

    def test_structure(self):
        """Has 63d and 126d windows."""
        df = _make_df()
        result = audit_rolling_stability(df)
        assert "63d" in result
        assert "126d" in result
        for wk in ["63d", "126d"]:
            for label in ["core", "vrp", "combined"]:
                assert label in result[wk]

    def test_rolling_values_plausible(self):
        """Mean rolling Sharpe should be in the right ballpark."""
        df = _make_df()
        result = audit_rolling_stability(df)
        # 63d rolling Sharpe mean should be finite
        mean_63 = result["63d"]["combined"]["mean"]
        assert mean_63 is not None
        assert np.isfinite(mean_63)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6 — PnL Concentration
# ═══════════════════════════════════════════════════════════════════════════

class TestPnLConcentration:

    def test_uniform_pnl_not_concentrated(self):
        """Equal daily PnL → low concentration."""
        np.random.seed(42)
        N = 200
        dates = pd.bdate_range("2023-01-01", periods=N)
        # All positive, equal PnL
        data = pd.DataFrame({
            "date": dates,
            "core_pnl": np.full(N, 100.0),
            "vrp_pnl": np.full(N, 50.0),
            "w_vrp": np.full(N, 0.10),
        })
        df = validate_and_load(data)
        result = audit_pnl_concentration(df)
        assert result["core"]["top5_pct"] < 5, "Uniform PnL should not be concentrated"

    def test_spike_pnl_concentrated(self):
        """One massive spike → concentrated."""
        np.random.seed(42)
        N = 200
        dates = pd.bdate_range("2023-01-01", periods=N)
        pnl = np.full(N, 10.0)
        pnl[50] = 100_000.0  # One huge day
        data = pd.DataFrame({
            "date": dates,
            "core_pnl": pnl,
            "vrp_pnl": np.full(N, 5.0),
            "w_vrp": np.full(N, 0.10),
        })
        df = validate_and_load(data)
        result = audit_pnl_concentration(df)
        assert result["core"]["concentrated"] is True

    def test_structure(self):
        """Has core, vrp, combined."""
        df = _make_df()
        result = audit_pnl_concentration(df)
        assert "core" in result
        assert "vrp" in result
        assert "combined" in result


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7 — OOS Segmentation
# ═══════════════════════════════════════════════════════════════════════════

class TestOOSSegmentation:

    def test_structure(self):
        """Has first/second counts and per-sleeve Sharpe."""
        df = _make_df()
        result = audit_oos_segmentation(df)
        assert "n_first" in result
        assert "n_second" in result
        for label in ["core", "vrp", "combined"]:
            assert "sharpe_first_half" in result[label]
            assert "sharpe_second_half" in result[label]

    def test_halves_correct_size(self):
        """Each half has ~N/2 observations."""
        df = _make_df(N=300)
        result = audit_oos_segmentation(df)
        assert result["n_first"] == 150
        assert result["n_second"] == 150

    def test_stable_series_not_regime_dependent(self):
        """IID returns → combined not regime-dependent."""
        df = _make_df()
        result = audit_oos_segmentation(df)
        # Combined portfolio should be stable; individual sleeves may diverge
        # with only 150 obs per half.
        assert result["combined"]["regime_dependent"] is False, \
            "combined falsely flagged as regime-dependent"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 8 — Integrity Classification
# ═══════════════════════════════════════════════════════════════════════════

class TestIntegrityClassification:

    def test_robust_when_no_flags(self):
        """No flags → ROBUST_HIGH_SHARPE."""
        result = classify_sharpe_integrity(
            autocorrelation={"combined": {"significant_autocorrelation": False}},
            utilization={"under_deployed_flag": False},
            cost={"baseline_sharpe": 2.0, "cost_scenarios": {"+10bps": {"sharpe": 1.9}}},
            bootstrap={},
            rolling={"63d": {"combined": {"unstable": False}}},
            concentration={"combined": {"concentrated": False}},
            oos={"combined": {"regime_dependent": False}},
        )
        assert result["classification"] == "ROBUST_HIGH_SHARPE"
        assert len(result["flags"]) == 0

    def test_vol_smoothing_detected(self):
        """Significant AC with large NW drop → VOLATILITY_SMOOTHING."""
        result = classify_sharpe_integrity(
            autocorrelation={"combined": {
                "significant_autocorrelation": True,
                "naive_sharpe": 4.0, "nw_sharpe_lag5": 2.5,
            }},
            utilization={"under_deployed_flag": False},
            cost={"baseline_sharpe": 4.0, "cost_scenarios": {"+10bps": {"sharpe": 3.8}}},
            bootstrap={},
            rolling={"63d": {"combined": {"unstable": False}}},
            concentration={"combined": {"concentrated": False}},
            oos={"combined": {"regime_dependent": False}},
        )
        assert result["classification"] == "LIKELY_VOLATILITY_SMOOTHING_ARTIFACT"
        assert "VOLATILITY_SMOOTHING" in result["flags"]

    def test_under_deployment_detected(self):
        """Under-deployed → UNDER_DEPLOYMENT_DRIVEN_SHARPE."""
        result = classify_sharpe_integrity(
            autocorrelation={"combined": {"significant_autocorrelation": False}},
            utilization={"under_deployed_flag": True,
                         "diagnosis": "Portfolio vol < 5%"},
            cost={"baseline_sharpe": 3.0, "cost_scenarios": {"+10bps": {"sharpe": 2.8}}},
            bootstrap={},
            rolling={"63d": {"combined": {"unstable": False}}},
            concentration={"combined": {"concentrated": False}},
            oos={"combined": {"regime_dependent": False}},
        )
        assert result["classification"] == "UNDER_DEPLOYMENT_DRIVEN_SHARPE"

    def test_cost_sensitive_detected(self):
        """Large Sharpe drop at +10bps → COST_SENSITIVE."""
        result = classify_sharpe_integrity(
            autocorrelation={"combined": {"significant_autocorrelation": False}},
            utilization={"under_deployed_flag": False},
            cost={"baseline_sharpe": 3.0, "cost_scenarios": {"+10bps": {"sharpe": 2.0}}},
            bootstrap={},
            rolling={"63d": {"combined": {"unstable": False}}},
            concentration={"combined": {"concentrated": False}},
            oos={"combined": {"regime_dependent": False}},
        )
        assert result["classification"] == "COST_SENSITIVE_SHARPE"

    def test_concentrated_detected(self):
        """Concentrated PnL → CONCENTRATED_ALPHA."""
        result = classify_sharpe_integrity(
            autocorrelation={"combined": {"significant_autocorrelation": False}},
            utilization={"under_deployed_flag": False},
            cost={"baseline_sharpe": 2.0, "cost_scenarios": {"+10bps": {"sharpe": 1.9}}},
            bootstrap={},
            rolling={"63d": {"combined": {"unstable": False}}},
            concentration={"combined": {"concentrated": True, "top5_pct": 45.0}},
            oos={"combined": {"regime_dependent": False}},
        )
        assert result["classification"] == "CONCENTRATED_ALPHA"


# ═══════════════════════════════════════════════════════════════════════════
# End-to-End
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEnd:

    def test_run_sharpe_audit_structure(self):
        """Full audit returns all 8 phase keys."""
        df = _make_df()
        report = run_sharpe_audit("e2e", df)
        expected = {
            "run_id", "n_observations",
            "autocorrelation_audit", "capital_utilization",
            "turnover_cost_sensitivity", "bootstrap_confidence",
            "rolling_stability", "pnl_concentration",
            "oos_segmentation", "integrity_diagnosis",
        }
        assert expected == set(report.keys())

    def test_deterministic(self):
        """Same input → same output."""
        df = _make_df(seed=77)
        r1 = run_sharpe_audit("d1", df)
        df2 = _make_df(seed=77)
        r2 = run_sharpe_audit("d2", df2)
        assert r1["integrity_diagnosis"]["classification"] == \
               r2["integrity_diagnosis"]["classification"]

    def test_classification_present(self):
        """Integrity diagnosis has classification."""
        df = _make_df()
        report = run_sharpe_audit("cls", df)
        diag = report["integrity_diagnosis"]
        assert "classification" in diag
        assert diag["classification"] in {
            "ROBUST_HIGH_SHARPE",
            "LIKELY_VOLATILITY_SMOOTHING_ARTIFACT",
            "UNDER_DEPLOYMENT_DRIVEN_SHARPE",
            "COST_SENSITIVE_SHARPE",
            "CONCENTRATED_ALPHA",
            "REGIME_DEPENDENT_SHARPE",
            "UNSTABLE_SHARPE",
            "INCONCLUSIVE",
        }
