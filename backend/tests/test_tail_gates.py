"""Tests for R6: tail-first validation gates."""
from __future__ import annotations

import pytest

from sleeves.cooc_reversal_futures.pipeline.types import TradeProxyReport
from sleeves.cooc_reversal_futures.pipeline.validation import _tail_risk_gates


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_report(**overrides) -> TradeProxyReport:
    """Build a synthetic TradeProxyReport with defaults."""
    defaults = dict(
        sharpe_model=1.5,
        sharpe_baseline=1.0,
        hit_rate=0.55,
        max_drawdown=-0.05,
        mean_daily_return=0.001,
        worst_1pct_return=-0.015,
        gate_passed=True,
        n_days=252,
        vol=0.01,
        skew=-0.2,
        kurtosis=3.0,
        cvar_1pct=-0.02,
        n_zero_return_days=0,
        n_insufficient_days=0,
    )
    defaults.update(overrides)
    return TradeProxyReport(**defaults)


# ---------------------------------------------------------------------------
# All-pass scenario
# ---------------------------------------------------------------------------

class TestTailGatesAllPass:
    def test_all_gates_pass(self):
        """Model clearly better than baseline → all gates pass."""
        model = _make_report(
            sharpe_model=1.8,
            max_drawdown=-0.03,
            cvar_1pct=-0.010,
            worst_1pct_return=-0.008,
            skew=0.1,
        )
        baseline = _make_report(
            sharpe_model=1.5,
            max_drawdown=-0.05,
            cvar_1pct=-0.020,
            worst_1pct_return=-0.015,
            skew=-0.2,
        )
        gates = _tail_risk_gates(model, baseline, delta_min=0.10)
        assert all(g.passed for g in gates), [g for g in gates if not g.passed]
        assert len(gates) == 5

    def test_gate_names(self):
        model = _make_report()
        baseline = _make_report(sharpe_model=1.0)
        gates = _tail_risk_gates(model, baseline)
        names = {g.name for g in gates}
        assert names == {
            "tail_sharpe_delta",
            "tail_maxdd",
            "tail_cvar_1pct",
            "tail_worst_1pct",
            "tail_skew",
        }


# ---------------------------------------------------------------------------
# Individual gate failures
# ---------------------------------------------------------------------------

class TestTailGateFailures:
    def test_sharpe_delta_fails(self):
        """Insufficient Sharpe improvement → hard fail."""
        model = _make_report(sharpe_model=1.05)
        baseline = _make_report(sharpe_model=1.0)
        gates = _tail_risk_gates(model, baseline, delta_min=0.10)
        sharpe_gate = [g for g in gates if g.name == "tail_sharpe_delta"][0]
        assert not sharpe_gate.passed

    def test_sharpe_delta_passes_at_boundary(self):
        """Exactly delta_min → pass."""
        model = _make_report(sharpe_model=1.10)
        baseline = _make_report(sharpe_model=1.0)
        gates = _tail_risk_gates(model, baseline, delta_min=0.10)
        sharpe_gate = [g for g in gates if g.name == "tail_sharpe_delta"][0]
        assert sharpe_gate.passed

    def test_maxdd_fails_when_worse(self):
        """Model drawdown worse than baseline beyond tolerance."""
        model = _make_report(max_drawdown=-0.10)
        baseline = _make_report(max_drawdown=-0.05)
        gates = _tail_risk_gates(model, baseline, dd_tolerance=0.02)
        dd_gate = [g for g in gates if g.name == "tail_maxdd"][0]
        assert not dd_gate.passed

    def test_maxdd_passes_within_tolerance(self):
        """Model DD slightly worse but within tolerance → pass."""
        model = _make_report(max_drawdown=-0.06)
        baseline = _make_report(max_drawdown=-0.05)
        gates = _tail_risk_gates(model, baseline, dd_tolerance=0.02)
        dd_gate = [g for g in gates if g.name == "tail_maxdd"][0]
        assert dd_gate.passed

    def test_cvar_fails_when_worse(self):
        """Model CVaR worse than baseline beyond tolerance."""
        model = _make_report(cvar_1pct=-0.05)
        baseline = _make_report(cvar_1pct=-0.02)
        gates = _tail_risk_gates(model, baseline, cvar_tolerance=0.005)
        cvar_gate = [g for g in gates if g.name == "tail_cvar_1pct"][0]
        assert not cvar_gate.passed

    def test_cvar_passes_within_tolerance(self):
        model = _make_report(cvar_1pct=-0.024)
        baseline = _make_report(cvar_1pct=-0.02)
        gates = _tail_risk_gates(model, baseline, cvar_tolerance=0.005)
        cvar_gate = [g for g in gates if g.name == "tail_cvar_1pct"][0]
        assert cvar_gate.passed

    def test_worst_1pct_fails_when_worse(self):
        """Soft gate: model's worst day worse than baseline's."""
        model = _make_report(worst_1pct_return=-0.03)
        baseline = _make_report(worst_1pct_return=-0.02)
        gates = _tail_risk_gates(model, baseline)
        w_gate = [g for g in gates if g.name == "tail_worst_1pct"][0]
        assert not w_gate.passed

    def test_skew_fails_when_worse(self):
        """Soft gate: model skew much worse than baseline."""
        model = _make_report(skew=-2.0)
        baseline = _make_report(skew=-0.2)
        gates = _tail_risk_gates(model, baseline, skew_tolerance=0.5)
        s_gate = [g for g in gates if g.name == "tail_skew"][0]
        assert not s_gate.passed

    def test_skew_passes_within_tolerance(self):
        model = _make_report(skew=-0.6)
        baseline = _make_report(skew=-0.2)
        gates = _tail_risk_gates(model, baseline, skew_tolerance=0.5)
        s_gate = [g for g in gates if g.name == "tail_skew"][0]
        assert s_gate.passed


# ---------------------------------------------------------------------------
# Isolation: only the targeted gate fails
# ---------------------------------------------------------------------------

class TestTailGateIsolation:
    def test_only_sharpe_fails(self):
        """When only Sharpe is bad, other gates pass."""
        model = _make_report(sharpe_model=1.0)  # no improvement
        baseline = _make_report(sharpe_model=1.0)
        gates = _tail_risk_gates(model, baseline, delta_min=0.10)
        failed = [g.name for g in gates if not g.passed]
        assert failed == ["tail_sharpe_delta"]

    def test_only_maxdd_fails(self):
        model = _make_report(sharpe_model=1.5, max_drawdown=-0.20)
        baseline = _make_report(sharpe_model=1.0, max_drawdown=-0.05)
        gates = _tail_risk_gates(model, baseline, delta_min=0.10, dd_tolerance=0.02)
        failed = [g.name for g in gates if not g.passed]
        assert failed == ["tail_maxdd"]
