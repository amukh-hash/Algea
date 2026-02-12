"""
Tests for validated backtest integration (sharpe_validator).

Covers:
  - Input validation (sorted, no dups, no NaN, NAV identity)
  - Weight timing check
  - Portfolio reconstruction
  - ES95
  - Economic diagnosis
  - End-to-end generate_validated_report
"""
import numpy as np
import pandas as pd
import pytest

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.analysis.sharpe_validator import (
    DataValidationError,
    validate_and_load,
    check_weight_timing,
    check_portfolio_reconstruction,
    compute_enhanced_sleeve_stats,
    compute_weight_diagnostics,
    classify_vrp_role,
    generate_validated_report,
    _compute_es,
)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _make_clean_data(N=200, seed=42, with_regime=True, with_portfolio_nav=False):
    """Deterministic synthetic daily data with no issues."""
    np.random.seed(seed)
    dates = pd.bdate_range("2023-01-01", periods=N)
    core_pnl = np.random.normal(500, 2000, N)
    vrp_pnl = np.random.normal(200, 1000, N)
    w_vrp = np.clip(np.random.uniform(0.05, 0.20, N), 0, 0.20)

    data = {
        "date": dates,
        "core_pnl": core_pnl,
        "vrp_pnl": vrp_pnl,
        "w_vrp": w_vrp,
    }

    if with_regime:
        regimes = np.random.choice(
            ["normal_carry", "caution", "crash_risk"], N, p=[0.6, 0.3, 0.1]
        )
        data["regime"] = regimes

    df = pd.DataFrame(data)

    if with_portfolio_nav:
        # Reconstruct NAV manually for a consistent test
        cap = 1_000_000.0
        core_nav = cap + np.cumsum(core_pnl)
        vrp_nav = cap + np.cumsum(vrp_pnl)

        core_nav_prev = np.empty(N)
        vrp_nav_prev = np.empty(N)
        core_nav_prev[0] = cap
        core_nav_prev[1:] = core_nav[:-1]
        vrp_nav_prev[0] = cap
        vrp_nav_prev[1:] = vrp_nav[:-1]

        core_ret = core_pnl / core_nav_prev
        vrp_ret = vrp_pnl / vrp_nav_prev
        port_ret = (1 - w_vrp) * core_ret + w_vrp * vrp_ret

        portfolio_nav = np.empty(N)
        portfolio_nav[0] = cap * (1 + port_ret[0])
        for i in range(1, N):
            portfolio_nav[i] = portfolio_nav[i - 1] * (1 + port_ret[i])

        df["portfolio_nav"] = portfolio_nav

    return df


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1 — Input Validation
# ═══════════════════════════════════════════════════════════════════════════

class TestInputValidation:
    """Strict input validation checks."""

    def test_clean_data_passes(self):
        """Clean data should validate without error."""
        data = _make_clean_data()
        df = validate_and_load(data)
        assert len(df) > 0
        assert "core_ret" in df.columns
        assert "port_ret" in df.columns

    def test_missing_columns_raises(self):
        """Missing required columns → DataValidationError."""
        data = pd.DataFrame({"date": pd.bdate_range("2023-01-01", periods=10)})
        with pytest.raises(DataValidationError, match="Missing required"):
            validate_and_load(data)

    def test_duplicate_dates_raises(self):
        """Duplicate dates → DataValidationError."""
        data = _make_clean_data(N=50)
        data = pd.concat([data, data.iloc[[5]]], ignore_index=True)
        with pytest.raises(DataValidationError, match="duplicate date"):
            validate_and_load(data)

    def test_nan_in_pnl_raises(self):
        """NaN in core_pnl → DataValidationError."""
        data = _make_clean_data(N=50)
        data.loc[10, "core_pnl"] = np.nan
        with pytest.raises(DataValidationError, match="NaN"):
            validate_and_load(data)

    def test_nan_in_w_vrp_raises(self):
        """NaN in w_vrp → DataValidationError."""
        data = _make_clean_data(N=50)
        data.loc[5, "w_vrp"] = np.nan
        with pytest.raises(DataValidationError, match="NaN"):
            validate_and_load(data)

    def test_unsorted_gets_sorted(self):
        """Unsorted dates are auto-sorted."""
        data = _make_clean_data(N=100)
        shuffled = data.sample(frac=1, random_state=7).reset_index(drop=True)
        df = validate_and_load(shuffled)
        assert df["date"].is_monotonic_increasing

    def test_nav_identity_violation_raises(self):
        """core_nav inconsistent with core_pnl → DataValidationError."""
        data = _make_clean_data(N=50)
        # Construct NAV and then corrupt it
        data["core_nav"] = 1_000_000 + data["core_pnl"].cumsum()
        data.loc[25, "core_nav"] += 9999  # large violation
        with pytest.raises(DataValidationError, match="NAV identity violated"):
            validate_and_load(data)

    def test_nav_identity_ok(self):
        """Correctly computed NAV passes identity check."""
        data = _make_clean_data(N=100)
        data["core_nav"] = 1_000_000 + data["core_pnl"].cumsum()
        data["vrp_nav"] = 1_000_000 + data["vrp_pnl"].cumsum()
        df = validate_and_load(data)
        assert len(df) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Phase 2 — Weight Timing Check
# ═══════════════════════════════════════════════════════════════════════════

class TestWeightTiming:
    """Weight timing bias detection."""

    def test_constant_weights_timing_ok(self):
        """Constant weights → current == lagged → timing OK."""
        data = _make_clean_data(N=200)
        data["w_vrp"] = 0.10  # constant
        df = validate_and_load(data)
        result = check_weight_timing(df)
        assert result["weight_timing_ok"] is True

    def test_variable_weights_returns_structure(self):
        """Variable weights produce full result structure."""
        data = _make_clean_data(N=200)
        df = validate_and_load(data)
        result = check_weight_timing(df)

        expected_keys = {
            "weight_timing_ok", "sharpe_current_weights", "sharpe_lagged_weights",
            "sharpe_pct_diff", "correlation", "mean_return_diff", "recommendation",
        }
        assert expected_keys == set(result.keys())

    def test_small_sample_skips(self):
        """< MIN_OBS observations → skip timing check gracefully."""
        data = _make_clean_data(N=25)
        df = validate_and_load(data)
        result = check_weight_timing(df)
        assert result["weight_timing_ok"] is True
        assert "Insufficient" in result.get("note", "")


# ═══════════════════════════════════════════════════════════════════════════
# Phase 3 — Portfolio Reconstruction
# ═══════════════════════════════════════════════════════════════════════════

class TestPortfolioReconstruction:
    """Portfolio NAV reconstruction verification."""

    def test_no_portfolio_nav_skips(self):
        """Without portfolio_nav column → skip with OK."""
        data = _make_clean_data(N=100)
        df = validate_and_load(data)
        result = check_portfolio_reconstruction(df)
        assert result["portfolio_reconstruction_ok"] is True
        assert "skipped" in result.get("note", "")

    def test_matching_nav_passes(self):
        """Correctly computed portfolio_nav passes reconstruction."""
        data = _make_clean_data(N=100, with_portfolio_nav=True)
        df = validate_and_load(data)
        # We need to pass original data that has portfolio_nav
        result = check_portfolio_reconstruction(data)
        assert result["portfolio_reconstruction_ok"] is True

    def test_corrupted_nav_raises(self):
        """Corrupted portfolio_nav → DataValidationError."""
        data = _make_clean_data(N=100, with_portfolio_nav=True)
        data.loc[50, "portfolio_nav"] += 1_000_000  # massive corruption
        with pytest.raises(DataValidationError, match="reconstruction failed"):
            check_portfolio_reconstruction(data)


# ═══════════════════════════════════════════════════════════════════════════
# Phase 4 — ES95
# ═══════════════════════════════════════════════════════════════════════════

class TestES95:
    """Expected Shortfall computation."""

    def test_es95_negative(self):
        """ES95 should be negative for normal returns with positive vol."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0003, 0.01, 500))
        es = _compute_es(returns, 0.95)
        assert es < 0, f"ES95 should be negative, got {es}"

    def test_es95_in_enhanced_stats(self):
        """Enhanced stats include es_95 key."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0003, 0.01, 200))
        stats = compute_enhanced_sleeve_stats(returns)
        assert "es_95" in stats
        assert isinstance(stats["es_95"], float)

    def test_es95_worse_than_percentile(self):
        """ES95 should be worse (more negative) than the 5th percentile."""
        np.random.seed(99)
        returns = pd.Series(np.random.normal(0.0, 0.015, 1000))
        es = _compute_es(returns, 0.95)
        p5 = float(returns.quantile(0.05))
        assert es <= p5, f"ES95 ({es}) should be <= 5th percentile ({p5})"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 5 — Weight Diagnostics
# ═══════════════════════════════════════════════════════════════════════════

class TestWeightDiagnostics:
    """Weight diagnostics by regime."""

    def test_mean_weight_correct(self):
        """Mean w_vrp matches manual computation."""
        data = _make_clean_data(N=200)
        df = validate_and_load(data)
        wd = compute_weight_diagnostics(df)
        expected = float(df["w_vrp"].mean())
        assert abs(wd["mean_w_vrp"] - expected) < 1e-5

    def test_regime_weights_present(self):
        """Per-regime weights should be present when regime column exists."""
        data = _make_clean_data(N=200, with_regime=True)
        df = validate_and_load(data)
        wd = compute_weight_diagnostics(df)
        assert "mean_w_vrp_by_regime" in wd
        assert "normal_carry" in wd["mean_w_vrp_by_regime"]

    def test_no_regime_no_breakdown(self):
        """Without regime → no by-regime breakdown."""
        data = _make_clean_data(N=100, with_regime=False)
        df = validate_and_load(data)
        wd = compute_weight_diagnostics(df)
        assert "mean_w_vrp_by_regime" not in wd


# ═══════════════════════════════════════════════════════════════════════════
# Phase 6 — Economic Diagnosis
# ═══════════════════════════════════════════════════════════════════════════

class TestEconomicDiagnosis:
    """Economic interpretation block."""

    def test_diversifying_alpha(self):
        """Positive VRP Sharpe + low correlation → diversifying alpha."""
        result = classify_vrp_role(
            core_sharpe=1.5, vrp_sharpe=0.8, combined_sharpe=1.7,
            correlation=0.1, mean_w_vrp=0.10,
        )
        assert result["classification"] == "diversifying_alpha_sleeve"

    def test_tail_hedge(self):
        """Near-zero VRP Sharpe + low correlation → tail hedge."""
        result = classify_vrp_role(
            core_sharpe=1.5, vrp_sharpe=0.05, combined_sharpe=1.4,
            correlation=0.1, mean_w_vrp=0.10,
        )
        assert result["classification"] == "tail_hedge_variance_dampener"

    def test_capital_inefficiency_flag(self):
        """Combined < Core → CAPITAL_INEFFICIENCY flag."""
        result = classify_vrp_role(
            core_sharpe=1.5, vrp_sharpe=-0.3, combined_sharpe=1.2,
            correlation=0.1, mean_w_vrp=0.10,
        )
        assert "CAPITAL_INEFFICIENCY" in result["flags"]

    def test_under_deployment_flag(self):
        """Mean w_vrp < 1% → UNDER_DEPLOYMENT flag."""
        result = classify_vrp_role(
            core_sharpe=1.5, vrp_sharpe=0.8, combined_sharpe=1.6,
            correlation=0.1, mean_w_vrp=0.005,
        )
        assert "UNDER_DEPLOYMENT" in result["flags"]

    def test_sharpe_improvement(self):
        """Combined > Core → sharpe_improvement_pct reported."""
        result = classify_vrp_role(
            core_sharpe=1.0, vrp_sharpe=0.8, combined_sharpe=1.2,
            correlation=0.1, mean_w_vrp=0.10,
        )
        assert "sharpe_improvement_pct" in result["details"]
        assert result["details"]["sharpe_improvement_pct"] == 20.0

    def test_correlated_alpha_sleeve(self):
        """VRP Sharpe > 0 but high correlation → correlated alpha."""
        result = classify_vrp_role(
            core_sharpe=1.5, vrp_sharpe=0.8, combined_sharpe=1.7,
            correlation=0.5, mean_w_vrp=0.10,
        )
        assert result["classification"] == "correlated_alpha_sleeve"


# ═══════════════════════════════════════════════════════════════════════════
# Phase 7 — End-to-End
# ═══════════════════════════════════════════════════════════════════════════

class TestEndToEnd:
    """Integration tests for generate_validated_report."""

    def test_report_structure(self):
        """Report has all expected top-level keys."""
        data = _make_clean_data(N=300)
        report = generate_validated_report("e2e_test", data)

        expected_keys = {
            "run_id", "core", "vrp", "combined",
            "correlation", "marginal_sharpe_contribution",
            "information_ratio", "diversification_benefit",
            "weight_diagnostics", "regime_breakdown",
            "validation_checks", "economic_diagnosis",
            "rolling_sharpe_summary",
        }
        assert expected_keys == set(report.keys())

    def test_validation_checks_present(self):
        """validation_checks has weight_timing and portfolio_reconstruction."""
        data = _make_clean_data(N=200)
        report = generate_validated_report("vc_test", data)
        checks = report["validation_checks"]
        assert "weight_timing" in checks
        assert "portfolio_reconstruction" in checks

    def test_es95_in_sleeves(self):
        """All sleeve stats have ES-95."""
        data = _make_clean_data(N=200)
        report = generate_validated_report("es_test", data)
        for sleeve in ["core", "vrp", "combined"]:
            assert "es_95" in report[sleeve]

    def test_economic_diagnosis_present(self):
        """Economic diagnosis has classification and flags."""
        data = _make_clean_data(N=200)
        report = generate_validated_report("econ_test", data)
        econ = report["economic_diagnosis"]
        assert "classification" in econ
        assert "flags" in econ
        assert isinstance(econ["flags"], list)

    def test_deterministic_sharpe(self):
        """Same seed → same Sharpe."""
        data1 = _make_clean_data(N=200, seed=77)
        data2 = _make_clean_data(N=200, seed=77)
        r1 = generate_validated_report("det1", data1)
        r2 = generate_validated_report("det2", data2)
        assert r1["core"]["sharpe"] == r2["core"]["sharpe"]
        assert r1["combined"]["sharpe"] == r2["combined"]["sharpe"]
