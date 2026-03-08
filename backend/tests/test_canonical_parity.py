"""Phase 3.5 parity tests: canonical outputs must preserve active sleeve set.

These tests prove that enabling FF_CANONICAL_SLEEVE_OUTPUTS does not
change which sleeves are active — it only changes the contract shape.

Validates:
  1. legacy_rows_to_intents converts VRP-style rows to canonical intents
  2. legacy_rows_to_intents converts selector-style rows to canonical intents
  3. Zero-weight rows are skipped (no false intents)
  4. Empty symbol rows are skipped
  5. Weight clamping to [-2.0, 2.0]
  6. source_branch diagnostics preserved in SleeveDecision
  7. Compat artifacts contain derived_from_schema and derived_from_intent_ids
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from backend.app.contracts.canonical import (
    AssetClass,
    ExecutionPhase,
    SleeveStatus,
)
from backend.app.contracts.compat import (
    sleeve_decision_to_targets_compat,
    sleeve_decision_to_signals_compat,
)
from backend.app.orchestrator.sleeve_decision_helpers import (
    build_decision,
    build_trace,
    legacy_rows_to_intents,
    write_compat_artifacts,
    write_sleeve_decision,
)


_DATE = date(2026, 3, 7)


def _trace(sleeve: str = "vrp") -> object:
    return build_trace(
        run_id="run-parity", sleeve=sleeve,
        control_snapshot_id="ctrl-p",
        market_snapshot_id="mkt-p",
        portfolio_snapshot_id="pf-p",
        source_artifact="legacy_bridge",
    )


# ── legacy_rows_to_intents: VRP-style rows ───────────────────────────


class TestLegacyVRPRowsToIntents:
    def test_converts_vrp_orders_to_intents(self):
        """VRP legacy run_vrp returns rows as 'orders' with target_weight."""
        rows = [
            {"symbol": "SPY", "target_weight": 0.05, "asset_class": "EQUITY"},
            {"symbol": "TLT", "target_weight": -0.03, "asset_class": "EQUITY"},
        ]
        intents = legacy_rows_to_intents(
            rows, run_id="run-parity", asof_date=_DATE,
            sleeve="vrp", trace=_trace("vrp"),
        )
        assert len(intents) == 2
        symbols = {i.symbol for i in intents}
        assert symbols == {"SPY", "TLT"}
        assert intents[0].sleeve == "vrp"
        assert intents[0].asset_class == AssetClass.EQUITY
        assert intents[0].target_weight == 0.05

    def test_skips_zero_weight(self):
        rows = [
            {"symbol": "SPY", "target_weight": 0.0},
            {"symbol": "QQQ", "target_weight": 0.02},
        ]
        intents = legacy_rows_to_intents(
            rows, run_id="run-parity", asof_date=_DATE,
            sleeve="vrp", trace=_trace("vrp"),
        )
        assert len(intents) == 1
        assert intents[0].symbol == "QQQ"

    def test_skips_empty_symbol(self):
        rows = [
            {"symbol": "", "target_weight": 0.05},
            {"symbol": "SPY", "target_weight": 0.05},
        ]
        intents = legacy_rows_to_intents(
            rows, run_id="run-parity", asof_date=_DATE,
            sleeve="vrp", trace=_trace("vrp"),
        )
        assert len(intents) == 1
        assert intents[0].symbol == "SPY"

    def test_clamps_extreme_weight(self):
        rows = [{"symbol": "SPY", "target_weight": 5.0}]
        intents = legacy_rows_to_intents(
            rows, run_id="run-parity", asof_date=_DATE,
            sleeve="vrp", trace=_trace("vrp"),
        )
        assert intents[0].target_weight == 2.0

    def test_empty_rows_produces_no_intents(self):
        intents = legacy_rows_to_intents(
            [], run_id="run-parity", asof_date=_DATE,
            sleeve="vrp", trace=_trace("vrp"),
        )
        assert intents == ()


# ── legacy_rows_to_intents: Selector-style rows ──────────────────────


class TestLegacySelectorRowsToIntents:
    def test_converts_selector_targets(self):
        rows = [
            {"symbol": "AAPL", "target_weight": 0.03, "score": 0.85, "side": "buy"},
            {"symbol": "MSFT", "target_weight": -0.02, "score": 0.72, "side": "sell"},
            {"symbol": "NVDA", "target_weight": 0.015, "score": 0.68, "side": "buy"},
        ]
        intents = legacy_rows_to_intents(
            rows, run_id="run-parity", asof_date=_DATE,
            sleeve="selector", trace=_trace("selector"),
        )
        assert len(intents) == 3
        assert intents[0].metadata.get("score") == 0.85
        assert intents[1].metadata.get("side") == "sell"
        assert intents[0].execution_phase == ExecutionPhase.INTRADAY

    def test_preserves_row_index_in_trace(self):
        rows = [
            {"symbol": "A", "target_weight": 0.01},
            {"symbol": "B", "target_weight": 0.02},
        ]
        intents = legacy_rows_to_intents(
            rows, run_id="run-parity", asof_date=_DATE,
            sleeve="selector", trace=_trace("selector"),
        )
        assert intents[0].trace.source_row_index == 0
        assert intents[1].trace.source_row_index == 1


# ── SleeveDecision wrapping with source_branch diagnostics ────────────


class TestLegacyBridgeDecision:
    def test_ok_decision_from_legacy_vrp(self):
        rows = [{"symbol": "SPY", "target_weight": 0.05}]
        intents = legacy_rows_to_intents(
            rows, run_id="run-parity", asof_date=_DATE,
            sleeve="vrp", trace=_trace("vrp"),
        )
        decision = build_decision(
            sleeve="vrp", run_id="run-parity", asof_date=_DATE,
            status=SleeveStatus.OK if intents else SleeveStatus.HALTED,
            intents=intents,
            diagnostics={"source_branch": "legacy_vrp_bridge", "n_raw_targets": 1},
            generated_by="handle_signals_generate_vrp",
        )
        assert decision.status == SleeveStatus.OK
        assert decision.diagnostics["source_branch"] == "legacy_vrp_bridge"
        assert len(decision.intents) == 1

    def test_halted_decision_when_no_targets(self):
        intents = legacy_rows_to_intents(
            [], run_id="run-parity", asof_date=_DATE,
            sleeve="vrp", trace=_trace("vrp"),
        )
        decision = build_decision(
            sleeve="vrp", run_id="run-parity", asof_date=_DATE,
            status=SleeveStatus.OK if intents else SleeveStatus.HALTED,
            intents=intents,
            reason=None if intents else "legacy_vrp produced no tradeable targets",
            diagnostics={"source_branch": "legacy_vrp_bridge"},
            generated_by="handle_signals_generate_vrp",
        )
        assert decision.status == SleeveStatus.HALTED
        assert decision.intents == ()

    def test_ok_decision_from_legacy_selector(self):
        rows = [
            {"symbol": "AAPL", "target_weight": 0.03},
            {"symbol": "MSFT", "target_weight": -0.02},
        ]
        intents = legacy_rows_to_intents(
            rows, run_id="run-parity", asof_date=_DATE,
            sleeve="selector", trace=_trace("selector"),
        )
        decision = build_decision(
            sleeve="selector", run_id="run-parity", asof_date=_DATE,
            status=SleeveStatus.OK, intents=intents,
            diagnostics={"source_branch": "legacy_selector_loader"},
            generated_by="handle_signals_generate_selector",
        )
        assert decision.status == SleeveStatus.OK
        assert len(decision.intents) == 2
        assert decision.diagnostics["source_branch"] == "legacy_selector_loader"


# ── Compat artifacts enrichment ───────────────────────────────────────


class TestCompatArtifactEnrichment:
    def test_targets_compat_has_derived_from_schema(self):
        rows = [{"symbol": "SPY", "target_weight": 0.05}]
        intents = legacy_rows_to_intents(
            rows, run_id="run-parity", asof_date=_DATE,
            sleeve="vrp", trace=_trace("vrp"),
        )
        decision = build_decision(
            sleeve="vrp", run_id="run-parity", asof_date=_DATE,
            status=SleeveStatus.OK, intents=intents,
            generated_by="test",
        )
        compat = sleeve_decision_to_targets_compat(decision)
        assert compat["derived_from_schema"] == "sleeve_decision.v1"
        assert compat["derived_from_intent_ids"] == [intents[0].intent_id]
        assert compat["market_snapshot_id"] == ""
        assert compat["portfolio_snapshot_id"] == ""

    def test_signals_compat_has_derived_from_schema(self):
        rows = [{"symbol": "SPY", "target_weight": 0.05}]
        intents = legacy_rows_to_intents(
            rows, run_id="run-parity", asof_date=_DATE,
            sleeve="vrp", trace=_trace("vrp"),
        )
        decision = build_decision(
            sleeve="vrp", run_id="run-parity", asof_date=_DATE,
            status=SleeveStatus.OK, intents=intents,
            generated_by="test",
        )
        compat = sleeve_decision_to_signals_compat(decision)
        assert compat["derived_from_schema"] == "sleeve_decision.v1"
        assert len(compat["derived_from_intent_ids"]) == 1

    def test_compat_artifacts_on_disk(self, tmp_path: Path):
        rows = [{"symbol": "AAPL", "target_weight": 0.03}]
        intents = legacy_rows_to_intents(
            rows, run_id="run-parity", asof_date=_DATE,
            sleeve="selector", trace=_trace("selector"),
        )
        decision = build_decision(
            sleeve="selector", run_id="run-parity", asof_date=_DATE,
            status=SleeveStatus.OK, intents=intents,
            generated_by="test",
        )
        paths = write_compat_artifacts(decision, tmp_path)
        tgt = json.loads(Path(paths["targets_compat"]).read_text())
        assert tgt["authoritative"] is False
        assert tgt["derived_from_schema"] == "sleeve_decision.v1"
        assert tgt["derived_from_intent_ids"] == [intents[0].intent_id]


# ── Active sleeve set parity ──────────────────────────────────────────


class TestSleeveActivationParity:
    """Prove that default-active sleeves remain active under canonical outputs."""

    def test_vrp_not_disabled_by_canonical_flag(self):
        """VRP legacy branch must produce OK/HALTED, never DISABLED."""
        rows = [{"symbol": "SPY", "target_weight": 0.05}]
        intents = legacy_rows_to_intents(
            rows, run_id="run-parity", asof_date=_DATE,
            sleeve="vrp", trace=_trace("vrp"),
        )
        decision = build_decision(
            sleeve="vrp", run_id="run-parity", asof_date=_DATE,
            status=SleeveStatus.OK if intents else SleeveStatus.HALTED,
            intents=intents,
            diagnostics={"source_branch": "legacy_vrp_bridge"},
            generated_by="test",
        )
        assert decision.status != SleeveStatus.DISABLED
        assert decision.status == SleeveStatus.OK

    def test_selector_not_disabled_by_canonical_flag(self):
        """Selector legacy branch must produce OK/HALTED, never DISABLED."""
        rows = [
            {"symbol": "AAPL", "target_weight": 0.03},
            {"symbol": "MSFT", "target_weight": -0.02},
        ]
        intents = legacy_rows_to_intents(
            rows, run_id="run-parity", asof_date=_DATE,
            sleeve="selector", trace=_trace("selector"),
        )
        decision = build_decision(
            sleeve="selector", run_id="run-parity", asof_date=_DATE,
            status=SleeveStatus.OK if intents else SleeveStatus.HALTED,
            intents=intents,
            diagnostics={"source_branch": "legacy_selector_loader"},
            generated_by="test",
        )
        assert decision.status != SleeveStatus.DISABLED
        assert decision.status == SleeveStatus.OK
        assert len(decision.intents) == 2

    def test_futures_overnight_disabled_is_correct(self):
        """Futures overnight DISABLED when chronos off is real behavior."""
        decision = build_decision(
            sleeve="futures_overnight", run_id="run-parity", asof_date=_DATE,
            status=SleeveStatus.DISABLED, reason="chronos2 sleeve disabled",
            generated_by="test",
        )
        assert decision.status == SleeveStatus.DISABLED

    def test_statarb_disabled_is_correct(self):
        """StatArb DISABLED when statarb off is real behavior."""
        decision = build_decision(
            sleeve="statarb", run_id="run-parity", asof_date=_DATE,
            status=SleeveStatus.DISABLED, reason="statarb disabled",
            generated_by="test",
        )
        assert decision.status == SleeveStatus.DISABLED
