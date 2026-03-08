"""Contract tests for canonical Intent Supremacy dataclasses.

Validates construction rules, field bounds, enum constraints,
invariant enforcement, and immutability on all canonical types.
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from backend.app.contracts.canonical import (
    AssetClass,
    ExecutionPhase,
    PlannedOrder,
    PositionDeltaPlan,
    RiskDecisionReport,
    SleeveDecision,
    SleeveStatus,
    TargetIntent,
    TraceRef,
    Violation,
)


# ── Fixtures ──────────────────────────────────────────────────────────

_NOW = datetime(2026, 3, 7, 10, 0, 0, tzinfo=timezone.utc)
_DATE = date(2026, 3, 7)


def _trace(**overrides) -> TraceRef:
    defaults = dict(
        run_id="run-001",
        sleeve="core",
        sleeve_run_id="run-001",
        source_artifact="test",
        source_row_index=0,
        control_snapshot_id="ctrl-001",
        market_snapshot_id="mkt-001",
        portfolio_snapshot_id="pf-001",
        policy_version="p.v1",
        model_version="m.v1",
        config_version="c.v1",
    )
    defaults.update(overrides)
    return TraceRef(**defaults)


def _intent(**overrides) -> TargetIntent:
    defaults = dict(
        intent_id="intent-001",
        run_id="run-001",
        asof_date=_DATE,
        sleeve="core",
        symbol="ES",
        asset_class=AssetClass.FUTURE,
        target_weight=0.02,
        execution_phase=ExecutionPhase.FUTURES_OPEN,
        multiplier=50.0,
        trace=_trace(),
    )
    defaults.update(overrides)
    return TargetIntent(**defaults)


def _decision(**overrides) -> SleeveDecision:
    defaults = dict(
        schema_version="sleeve_decision.v1",
        sleeve="core",
        run_id="run-001",
        asof_date=_DATE,
        status=SleeveStatus.OK,
        reason=None,
        intents=(_intent(),),
        diagnostics={},
        warnings=(),
        artifact_refs={},
        control_snapshot_id="ctrl-001",
        market_snapshot_id="mkt-001",
        portfolio_snapshot_id="pf-001",
        started_at=_NOW,
        completed_at=_NOW,
        generated_by="test",
    )
    defaults.update(overrides)
    return SleeveDecision(**defaults)


# ── TraceRef ──────────────────────────────────────────────────────────


class TestTraceRef:
    def test_valid_construction(self):
        t = _trace()
        assert t.sleeve == "core"
        assert t.control_snapshot_id == "ctrl-001"
        assert t.run_id == "run-001"

    def test_frozen(self):
        t = _trace()
        with pytest.raises(AttributeError):
            t.sleeve = "vrp"  # type: ignore[misc]

    def test_lineage_fields_present(self):
        t = _trace()
        assert t.market_snapshot_id == "mkt-001"
        assert t.portfolio_snapshot_id == "pf-001"
        assert t.model_version == "m.v1"


# ── TargetIntent ──────────────────────────────────────────────────────


class TestTargetIntent:
    def test_valid_construction(self):
        ti = _intent()
        assert ti.symbol == "ES"
        assert ti.target_weight == 0.02
        assert ti.asset_class == AssetClass.FUTURE
        assert ti.run_id == "run-001"

    def test_weight_too_high(self):
        with pytest.raises(ValueError, match="target_weight"):
            _intent(target_weight=3.0)

    def test_weight_too_low(self):
        with pytest.raises(ValueError, match="target_weight"):
            _intent(target_weight=-3.0)

    def test_zero_multiplier(self):
        with pytest.raises(ValueError, match="multiplier"):
            _intent(multiplier=0.0)

    def test_negative_multiplier(self):
        with pytest.raises(ValueError, match="multiplier"):
            _intent(multiplier=-1.0)

    def test_empty_symbol(self):
        with pytest.raises(ValueError, match="symbol"):
            _intent(symbol="")

    def test_whitespace_symbol(self):
        with pytest.raises(ValueError, match="symbol"):
            _intent(symbol="   ")

    def test_empty_intent_id(self):
        with pytest.raises(ValueError, match="intent_id"):
            _intent(intent_id="")

    def test_boundary_weight_positive(self):
        ti = _intent(target_weight=2.0)
        assert ti.target_weight == 2.0

    def test_boundary_weight_negative(self):
        ti = _intent(target_weight=-2.0)
        assert ti.target_weight == -2.0

    def test_frozen(self):
        ti = _intent()
        with pytest.raises(AttributeError):
            ti.target_weight = 0.5  # type: ignore[misc]


# ── SleeveDecision — 4-state invariant table ──────────────────────────


class TestSleeveDecision:
    def test_valid_ok(self):
        d = _decision()
        assert d.status == SleeveStatus.OK
        assert len(d.intents) == 1
        assert d.run_id == "run-001"

    def test_ok_requires_intents(self):
        with pytest.raises(ValueError, match="status=OK but intents is empty"):
            _decision(intents=())

    def test_halted_requires_empty_intents(self):
        d = _decision(status=SleeveStatus.HALTED, intents=(), reason="no edge")
        assert d.status == SleeveStatus.HALTED
        assert d.intents == ()

    def test_halted_forbids_intents(self):
        with pytest.raises(ValueError, match="status=HALTED but intents is non-empty"):
            _decision(status=SleeveStatus.HALTED)

    def test_halted_allows_diagnostics_and_warnings(self):
        d = _decision(
            status=SleeveStatus.HALTED,
            intents=(),
            reason="turnover exhausted",
            diagnostics={"turnover_used": 0.95},
            warnings=("approaching daily limit",),
        )
        assert d.diagnostics["turnover_used"] == 0.95
        assert len(d.warnings) == 1

    def test_failed_forbids_intents(self):
        with pytest.raises(ValueError, match="status=FAILED but intents is non-empty"):
            _decision(status=SleeveStatus.FAILED)

    def test_failed_with_no_intents(self):
        d = _decision(status=SleeveStatus.FAILED, intents=(), reason="data unavailable")
        assert d.status == SleeveStatus.FAILED
        assert d.reason == "data unavailable"

    def test_disabled_forbids_intents(self):
        with pytest.raises(ValueError, match="status=DISABLED but intents is non-empty"):
            _decision(status=SleeveStatus.DISABLED)

    def test_disabled_with_no_intents(self):
        d = _decision(status=SleeveStatus.DISABLED, intents=(), reason="feature off")
        assert d.status == SleeveStatus.DISABLED

    def test_schema_version_literal(self):
        d = _decision()
        assert d.schema_version == "sleeve_decision.v1"

    def test_snapshot_ids_present(self):
        d = _decision()
        assert d.control_snapshot_id == "ctrl-001"
        assert d.market_snapshot_id == "mkt-001"
        assert d.portfolio_snapshot_id == "pf-001"


# ── Violation ─────────────────────────────────────────────────────────


class TestViolation:
    def test_error_severity(self):
        v = Violation(code="GROSS_LIMIT", message="exceeded", severity="error")
        assert v.severity == "error"

    def test_warning_severity(self):
        v = Violation(code="TURNOVER_HIGH", message="high turnover", severity="warning")
        assert v.severity == "warning"


# ── RiskDecisionReport ────────────────────────────────────────────────


class TestRiskDecisionReport:
    def test_valid_ok(self):
        r = RiskDecisionReport(
            schema_version="risk_decision.v2",
            decision_id="risk-001",
            run_id="run-001",
            asof_date=_DATE,
            status="ok",
            control_snapshot_id="ctrl-001",
            market_snapshot_id="mkt-001",
            portfolio_snapshot_id="pf-001",
            input_family="canonical_intents",
            source_sleeves=("core", "vrp"),
            violations=(),
            exposures={"gross": 0.5},
            limits={"max_gross": 1.5},
            diagnostics={},
            generated_by="risk_engine_v2",
        )
        assert r.status == "ok"
        assert len(r.violations) == 0
        assert r.run_id == "run-001"


# ── PlannedOrder ──────────────────────────────────────────────────────


class TestPlannedOrder:
    def test_valid(self):
        o = PlannedOrder(
            order_id="ord-001",
            symbol="SPY",
            side="BUY",
            qty=10,
            est_price=500.0,
            est_notional=5000.0,
            intent_refs=("intent-001",),
            price_source="broker",
        )
        assert o.qty == 10

    def test_zero_qty_rejected(self):
        with pytest.raises(ValueError, match="qty must be positive"):
            PlannedOrder(
                order_id="ord-002",
                symbol="SPY",
                side="BUY",
                qty=0,
                est_price=500.0,
                est_notional=0.0,
                intent_refs=(),
                price_source="broker",
            )

    def test_negative_price_rejected(self):
        with pytest.raises(ValueError, match="est_price must be non-negative"):
            PlannedOrder(
                order_id="ord-003",
                symbol="SPY",
                side="SELL",
                qty=5,
                est_price=-1.0,
                est_notional=-5.0,
                intent_refs=(),
                price_source="broker",
            )
