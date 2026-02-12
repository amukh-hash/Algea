"""Tests for empirical validation suite (Phase 1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from backend.app.evaluation.vrp_validation import (
    AllocationAnalysis,
    RegimeStats,
    StressWindowResult,
    ValidationSummary,
    analyze_allocation_stability,
    compute_es,
    compute_pnl_by_regime,
    compute_regime_durations,
    compute_regime_frequency,
    compute_transition_matrix,
    run_validation,
    validate_stress_window,
)


def _make_test_data(n: int = 100):
    """Create synthetic test data."""
    dates = pd.bdate_range("2024-01-01", periods=n)
    regimes = pd.Series(
        ["normal_carry"] * 60 + ["caution"] * 25 + ["crash_risk"] * 15,
        index=dates[:n],
    )
    daily_pnl = pd.Series(
        np.concatenate([
            np.random.normal(0.001, 0.005, 60),
            np.random.normal(-0.001, 0.008, 25),
            np.random.normal(-0.005, 0.015, 15),
        ]),
        index=dates[:n],
    )
    scenario = pd.Series(np.random.uniform(0.01, 0.05, n), index=dates[:n])
    w_vrp = pd.Series(np.random.uniform(0.05, 0.20, n), index=dates[:n])
    health = pd.Series(np.random.uniform(0.70, 0.95, n), index=dates[:n])
    dz = pd.Series(np.random.randint(0, 2, n), index=dates[:n])
    dr = pd.Series(np.random.randint(0, 3, n), index=dates[:n])
    return regimes, daily_pnl, scenario, w_vrp, health, dz, dr


class TestRegimeFrequency:
    def test_sums_to_one(self):
        regimes = pd.Series(["normal_carry", "caution", "crash_risk"] * 10)
        freq = compute_regime_frequency(regimes)
        assert abs(sum(freq.values()) - 1.0) < 1e-8

    def test_single_regime(self):
        regimes = pd.Series(["normal_carry"] * 20)
        freq = compute_regime_frequency(regimes)
        assert freq["normal_carry"] == 1.0


class TestRegimeDurations:
    def test_constant_regime(self):
        regimes = pd.Series(["caution"] * 10)
        durations = compute_regime_durations(regimes)
        assert durations["caution"] == 10.0

    def test_alternating(self):
        regimes = pd.Series(["normal_carry", "caution"] * 5)
        durations = compute_regime_durations(regimes)
        assert durations["normal_carry"] == 1.0
        assert durations["caution"] == 1.0


class TestTransitionMatrix:
    def test_no_self_transitions_alternating(self):
        regimes = pd.Series(["normal_carry", "caution"] * 5)
        matrix = compute_transition_matrix(regimes)
        assert matrix["normal_carry"]["caution"] == 5
        assert matrix["caution"]["normal_carry"] == 4


class TestES:
    def test_negative_es(self):
        pnl = pd.Series(np.random.normal(-0.01, 0.02, 100))
        es = compute_es(pnl, 0.95)
        assert es < 0  # expected shortfall of negative distribution

    def test_short_series_returns_zero(self):
        es = compute_es(pd.Series([1.0, 2.0]), 0.95)
        assert es == 0.0


class TestAllocationStability:
    def test_zero_churn_for_constant(self):
        w = pd.Series([0.10] * 50)
        analysis = analyze_allocation_stability(w)
        assert analysis.churn_days == 0
        assert analysis.max_daily_change < 1e-8

    def test_high_churn_detected(self):
        w = pd.Series([0.0, 0.10, 0.0, 0.10, 0.0] * 10)
        analysis = analyze_allocation_stability(w, churn_threshold=0.05)
        assert analysis.churn_days > 0


class TestStressWindow:
    def test_missing_data_returns_zero(self):
        regimes = pd.Series(["normal_carry"] * 10,
                            index=pd.bdate_range("2024-01-01", periods=10))
        losses = pd.Series([0.01] * 10, index=regimes.index)
        result = validate_stress_window(regimes, losses, 0.06,
                                        "test", "2020-02-15", "2020-04-15")
        assert result.total_days == 0
        assert not result.crash_risk_triggered


class TestFullValidation:
    def test_run_produces_summary(self):
        regimes, pnl, scenario, w, health, dz, dr = _make_test_data()
        summary = run_validation(regimes, pnl, scenario, w, health, dz, dr)
        assert isinstance(summary, ValidationSummary)
        assert len(summary.regime_stats) == 3
        assert summary.total_days == 100

    def test_compliance_bounded(self):
        regimes, pnl, scenario, w, health, dz, dr = _make_test_data()
        summary = run_validation(regimes, pnl, scenario, w, health, dz, dr)
        assert 0.0 <= summary.scenario_budget_compliance_pct <= 1.0
