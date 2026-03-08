"""Tests for sleeve_decision_helpers — Phase 3 canonical emission utilities.

Validates:
  - build_trace produces valid TraceRef
  - build_intent produces valid TargetIntent with auto-generated ID
  - build_decision produces valid SleeveDecision across all 4 states
  - write_sleeve_decision writes JSON to sleeves/<name>/decision.json
  - write_compat_artifacts writes legacy compat with authoritative=false
  - get_snapshot_ids extracts typed snapshot IDs from context
"""
from __future__ import annotations

import json
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from backend.app.contracts.canonical import (
    AssetClass,
    ExecutionPhase,
    SleeveDecision,
    SleeveStatus,
    TargetIntent,
)
from backend.app.orchestrator.sleeve_decision_helpers import (
    build_decision,
    build_intent,
    build_trace,
    get_snapshot_ids,
    write_compat_artifacts,
    write_sleeve_decision,
)


_DATE = date(2026, 3, 7)


def _trace(**kw):
    defaults = dict(
        run_id="run-001",
        sleeve="core",
        control_snapshot_id="ctrl-001",
        market_snapshot_id="mkt-001",
        portfolio_snapshot_id="pf-001",
        source_artifact="test",
    )
    defaults.update(kw)
    return build_trace(**defaults)


class TestBuildTrace:
    def test_valid(self):
        t = _trace()
        assert t.sleeve == "core"
        assert t.run_id == "run-001"
        assert t.control_snapshot_id == "ctrl-001"

    def test_frozen(self):
        t = _trace()
        with pytest.raises(AttributeError):
            t.run_id = "other"  # type: ignore[misc]


class TestBuildIntent:
    def test_valid(self):
        trace = _trace()
        intent = build_intent(
            run_id="run-001", asof_date=_DATE, sleeve="core",
            symbol="ES", asset_class=AssetClass.FUTURE,
            target_weight=0.02, execution_phase=ExecutionPhase.FUTURES_OPEN,
            multiplier=50.0, trace=trace,
        )
        assert intent.symbol == "ES"
        assert intent.intent_id  # auto-generated
        assert intent.run_id == "run-001"

    def test_auto_id(self):
        trace = _trace()
        i1 = build_intent(
            run_id="run-001", asof_date=_DATE, sleeve="core",
            symbol="ES", asset_class=AssetClass.FUTURE,
            target_weight=0.02, execution_phase=ExecutionPhase.FUTURES_OPEN,
            multiplier=50.0, trace=trace,
        )
        i2 = build_intent(
            run_id="run-001", asof_date=_DATE, sleeve="core",
            symbol="ES", asset_class=AssetClass.FUTURE,
            target_weight=0.02, execution_phase=ExecutionPhase.FUTURES_OPEN,
            multiplier=50.0, trace=trace,
        )
        assert i1.intent_id != i2.intent_id


class TestBuildDecision:
    def test_ok(self):
        trace = _trace()
        intent = build_intent(
            run_id="run-001", asof_date=_DATE, sleeve="core",
            symbol="ES", asset_class=AssetClass.FUTURE,
            target_weight=0.02, execution_phase=ExecutionPhase.FUTURES_OPEN,
            multiplier=50.0, trace=trace,
        )
        d = build_decision(
            sleeve="core", run_id="run-001", asof_date=_DATE,
            status=SleeveStatus.OK, intents=(intent,),
            generated_by="test",
        )
        assert d.status == SleeveStatus.OK
        assert len(d.intents) == 1

    def test_halted(self):
        d = build_decision(
            sleeve="core", run_id="run-001", asof_date=_DATE,
            status=SleeveStatus.HALTED, reason="no edge",
            generated_by="test",
        )
        assert d.status == SleeveStatus.HALTED
        assert d.intents == ()

    def test_failed(self):
        d = build_decision(
            sleeve="core", run_id="run-001", asof_date=_DATE,
            status=SleeveStatus.FAILED, reason="data error",
            generated_by="test",
        )
        assert d.status == SleeveStatus.FAILED

    def test_disabled(self):
        d = build_decision(
            sleeve="futures_overnight", run_id="run-001", asof_date=_DATE,
            status=SleeveStatus.DISABLED, reason="feature off",
            generated_by="test",
        )
        assert d.status == SleeveStatus.DISABLED


class TestWriteSleeveDecision:
    def test_writes_json(self, tmp_path: Path):
        trace = _trace()
        intent = build_intent(
            run_id="run-001", asof_date=_DATE, sleeve="core",
            symbol="ES", asset_class=AssetClass.FUTURE,
            target_weight=0.02, execution_phase=ExecutionPhase.FUTURES_OPEN,
            multiplier=50.0, trace=trace,
        )
        d = build_decision(
            sleeve="core", run_id="run-001", asof_date=_DATE,
            status=SleeveStatus.OK, intents=(intent,),
            generated_by="test",
        )
        paths = write_sleeve_decision(d, tmp_path)
        assert "decision" in paths
        decision_path = Path(paths["decision"])
        assert decision_path.exists()
        data = json.loads(decision_path.read_text())
        assert data["status"] == "ok"
        assert data["sleeve"] == "core"
        assert len(data["intents"]) == 1


class TestWriteCompatArtifacts:
    def test_writes_with_authoritative_false(self, tmp_path: Path):
        d = build_decision(
            sleeve="vrp", run_id="run-002", asof_date=_DATE,
            status=SleeveStatus.HALTED, reason="no edge",
            generated_by="test",
        )
        paths = write_compat_artifacts(d, tmp_path)
        tgt = json.loads(Path(paths["targets_compat"]).read_text())
        sig = json.loads(Path(paths["signals_compat"]).read_text())
        assert tgt["authoritative"] is False
        assert sig["authoritative"] is False
        assert tgt["derived_from"] == "canonical_intents"
        assert sig["derived_from"] == "canonical_intents"


class TestGetSnapshotIds:
    def test_from_typed_snapshots(self):
        from backend.app.contracts.providers import ControlSnapshot, MarketDataSnapshot, PortfolioStateSnapshot

        ctrl = ControlSnapshot(
            snapshot_id="ctrl-x", asof_date=_DATE, paused=False,
            execution_mode="paper", blocked_symbols=frozenset(),
            frozen_sleeves=frozenset(), gross_cap=1.5,
        )
        mkt = MarketDataSnapshot(
            snapshot_id="mkt-x", asof_date=_DATE,
            created_at=datetime.now(timezone.utc),
            quotes={}, historical_bars={}, option_surfaces={},
            feature_frames={}, freshness={},
        )
        pf = PortfolioStateSnapshot(
            snapshot_id="pf-x", asof_date=_DATE,
            created_at=datetime.now(timezone.utc),
            positions={},
        )
        ctx = {
            "typed_control_snapshot": ctrl,
            "typed_market_snapshot": mkt,
            "typed_portfolio_snapshot": pf,
        }
        c, m, p = get_snapshot_ids(ctx)
        assert c == "ctrl-x"
        assert m == "mkt-x"
        assert p == "pf-x"

    def test_fallback_when_no_typed(self):
        ctx = {"control_snapshot_id": "ctrl-legacy"}
        c, m, p = get_snapshot_ids(ctx)
        assert c == "ctrl-legacy"
        assert m == ""
        assert p == ""
