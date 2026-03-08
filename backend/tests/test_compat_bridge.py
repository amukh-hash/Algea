"""Tests for the canonical-to-legacy compatibility bridge.

Validates:
  - Derived artifacts always include ``authoritative: false``
  - Derived artifacts always include ``derived_from``
  - Derived artifacts always include ``source_run_id``
  - Compat derivation preserves lineage refs
  - Pydantic→canonical conversion fails on invalid inputs
  - Compat bridge does not silently drop important fields
"""
from __future__ import annotations

from datetime import date, datetime, timezone

import pytest

from backend.app.contracts.canonical import (
    AssetClass,
    ExecutionPhase,
    SleeveDecision,
    SleeveStatus,
    TargetIntent,
    TraceRef,
)
from backend.app.contracts.compat import (
    intents_to_targets_compat,
    pydantic_intent_to_canonical,
    sleeve_decision_to_signals_compat,
    sleeve_decision_to_targets_compat,
)
from backend.app.contracts.errors import ContractViolationError


_NOW = datetime(2026, 3, 7, 10, 0, 0, tzinfo=timezone.utc)
_DATE = date(2026, 3, 7)


def _trace() -> TraceRef:
    return TraceRef(
        run_id="run-001",
        sleeve="vrp",
        sleeve_run_id="run-001",
        source_artifact="test",
        source_row_index=0,
        control_snapshot_id="ctrl-001",
        market_snapshot_id="mkt-001",
        portfolio_snapshot_id="pf-001",
    )


def _decision() -> SleeveDecision:
    return SleeveDecision(
        schema_version="sleeve_decision.v1",
        sleeve="vrp",
        run_id="run-001",
        asof_date=_DATE,
        status=SleeveStatus.OK,
        reason=None,
        intents=(
            TargetIntent(
                intent_id="ti-1",
                run_id="run-001",
                asof_date=_DATE,
                sleeve="vrp",
                symbol="SPY",
                asset_class=AssetClass.EQUITY,
                target_weight=0.05,
                execution_phase=ExecutionPhase.INTRADAY,
                multiplier=1.0,
                trace=_trace(),
                metadata={"tenor": 30},
            ),
            TargetIntent(
                intent_id="ti-2",
                run_id="run-001",
                asof_date=_DATE,
                sleeve="vrp",
                symbol="QQQ",
                asset_class=AssetClass.EQUITY,
                target_weight=-0.03,
                execution_phase=ExecutionPhase.INTRADAY,
                multiplier=1.0,
                trace=_trace(),
            ),
        ),
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


# ── Targets Compat ────────────────────────────────────────────────────


class TestSleeveDecisionToTargetsCompat:
    def test_basic_structure(self):
        result = sleeve_decision_to_targets_compat(_decision())
        assert result["schema_version"] == "targets_compat.v1"
        assert result["derived_from"] == "canonical_intents"
        assert result["authoritative"] is False
        assert result["sleeve"] == "vrp"
        assert len(result["targets"]) == 2

    def test_contains_source_run_id(self):
        result = sleeve_decision_to_targets_compat(_decision())
        assert result["source_run_id"] == "run-001"

    def test_contains_control_snapshot_id(self):
        result = sleeve_decision_to_targets_compat(_decision())
        assert result["control_snapshot_id"] == "ctrl-001"

    def test_target_fields(self):
        result = sleeve_decision_to_targets_compat(_decision())
        t = result["targets"][0]
        assert t["symbol"] == "SPY"
        assert t["target_weight"] == 0.05
        assert t["asset_class"] == "EQUITY"
        assert t["execution_phase"] == "intraday"
        assert t["intent_id"] == "ti-1"

    def test_asof_date_present(self):
        result = sleeve_decision_to_targets_compat(_decision())
        assert result["asof_date"] == "2026-03-07"


# ── Signals Compat ────────────────────────────────────────────────────


class TestSleeveDecisionToSignalsCompat:
    def test_basic_structure(self):
        result = sleeve_decision_to_signals_compat(_decision())
        assert result["schema_version"] == "signals_compat.v1"
        assert result["authoritative"] is False
        assert result["source"] == "canonical_sleeve_decision"
        assert len(result["signals"]) == 2

    def test_contains_source_run_id(self):
        result = sleeve_decision_to_signals_compat(_decision())
        assert result["source_run_id"] == "run-001"

    def test_signal_score_matches_weight(self):
        result = sleeve_decision_to_signals_compat(_decision())
        assert result["signals"][0]["score"] == 0.05
        assert result["signals"][1]["score"] == -0.03

    def test_signal_preserves_intent_id(self):
        result = sleeve_decision_to_signals_compat(_decision())
        assert result["signals"][0]["intent_id"] == "ti-1"


# ── Combined Intents Compat ───────────────────────────────────────────


class TestIntentsToTargetsCompat:
    def test_groups_by_sleeve(self):
        d = _decision()
        result = intents_to_targets_compat(d.intents, run_id="run-001")
        assert "vrp" in result["by_sleeve"]
        assert len(result["by_sleeve"]["vrp"]) == 2
        assert result["total_intents"] == 2
        assert result["authoritative"] is False
        assert result["source_run_id"] == "run-001"


# ── Pydantic Bridge — Valid ───────────────────────────────────────────


class TestPydanticIntentToCanonical:
    def test_bridge_converts_fields(self):
        from backend.app.core.schemas import TargetIntent as PydanticTI
        from backend.app.core.schemas import ExecutionPhase as PydanticPhase

        pydantic_ti = PydanticTI(
            asof_date="2026-03-07",
            sleeve="core",
            symbol="ES",
            asset_class="FUTURE",
            target_weight=0.02,
            execution_phase=PydanticPhase.FUTURES_OPEN,
            multiplier=50.0,
            dte=-1,
        )
        canonical = pydantic_intent_to_canonical(pydantic_ti, run_id="run-bridge")
        assert canonical.sleeve == "core"
        assert canonical.symbol == "ES"
        assert canonical.asset_class == AssetClass.FUTURE
        assert canonical.target_weight == 0.02
        assert canonical.execution_phase == ExecutionPhase.FUTURES_OPEN
        assert canonical.multiplier == 50.0
        assert canonical.run_id == "run-bridge"
        assert canonical.trace.sleeve == "core"
        assert canonical.trace.run_id == "run-bridge"
        assert canonical.metadata.get("dte") == -1

    def test_bridge_auto_generates_intent_id(self):
        from backend.app.core.schemas import TargetIntent as PydanticTI

        pydantic_ti = PydanticTI(
            asof_date="2026-03-07",
            sleeve="vrp",
            symbol="SPY",
            asset_class="EQUITY",
            target_weight=0.05,
            execution_phase="intraday",
            multiplier=1.0,
        )
        canonical = pydantic_intent_to_canonical(pydantic_ti)
        assert canonical.intent_id
        assert len(canonical.intent_id) > 0

    def test_bridge_with_explicit_trace(self):
        from backend.app.core.schemas import TargetIntent as PydanticTI

        pydantic_ti = PydanticTI(
            asof_date="2026-03-07",
            sleeve="selector",
            symbol="AAPL",
            asset_class="EQUITY",
            target_weight=0.03,
            execution_phase="intraday",
            multiplier=1.0,
        )
        trace = _trace()
        canonical = pydantic_intent_to_canonical(pydantic_ti, trace=trace)
        assert canonical.trace.control_snapshot_id == "ctrl-001"


# ── Pydantic Bridge — Invalid Inputs (Modification 5) ────────────────


class TestPydanticBridgeInvalidInputs:
    def test_invalid_asset_class_raises_contract_violation(self):
        """Unknown asset class must raise ContractViolationError."""
        from unittest.mock import MagicMock
        fake = MagicMock()
        fake.asof_date = "2026-03-07"
        fake.sleeve = "core"
        fake.symbol = "XYZ"
        fake.asset_class = "CRYPTO"  # not in AssetClass enum
        fake.target_weight = 0.01
        fake.execution_phase = "intraday"
        fake.multiplier = 1.0
        fake.dte = -1
        with pytest.raises(ContractViolationError, match="asset_class"):
            pydantic_intent_to_canonical(fake)

    def test_invalid_execution_phase_raises_contract_violation(self):
        from unittest.mock import MagicMock
        fake = MagicMock()
        fake.asof_date = "2026-03-07"
        fake.sleeve = "core"
        fake.symbol = "SPY"
        fake.asset_class = "EQUITY"
        fake.target_weight = 0.01
        fake.execution_phase = "midnight_auction"  # not in ExecutionPhase
        fake.multiplier = 1.0
        fake.dte = -1
        with pytest.raises(ContractViolationError, match="execution_phase"):
            pydantic_intent_to_canonical(fake)

    def test_empty_symbol_raises_contract_violation(self):
        from unittest.mock import MagicMock
        fake = MagicMock()
        fake.asof_date = "2026-03-07"
        fake.sleeve = "core"
        fake.symbol = "   "  # whitespace only
        fake.asset_class = "EQUITY"
        fake.target_weight = 0.01
        fake.execution_phase = "intraday"
        fake.multiplier = 1.0
        fake.dte = -1
        with pytest.raises(ContractViolationError, match="empty symbol"):
            pydantic_intent_to_canonical(fake)

    def test_non_positive_multiplier_raises_contract_violation(self):
        from unittest.mock import MagicMock
        fake = MagicMock()
        fake.asof_date = "2026-03-07"
        fake.sleeve = "core"
        fake.symbol = "SPY"
        fake.asset_class = "EQUITY"
        fake.target_weight = 0.01
        fake.execution_phase = "intraday"
        fake.multiplier = 0.0  # invalid
        fake.dte = -1
        with pytest.raises(ContractViolationError, match="multiplier"):
            pydantic_intent_to_canonical(fake)

    def test_weight_out_of_range_raises_contract_violation(self):
        from unittest.mock import MagicMock
        fake = MagicMock()
        fake.asof_date = "2026-03-07"
        fake.sleeve = "core"
        fake.symbol = "SPY"
        fake.asset_class = "EQUITY"
        fake.target_weight = 5.0  # way out of [-2.0, 2.0]
        fake.execution_phase = "intraday"
        fake.multiplier = 1.0
        fake.dte = -1
        with pytest.raises(ContractViolationError, match="target_weight"):
            pydantic_intent_to_canonical(fake)
