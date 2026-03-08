"""Phase 4.1 hardening tests for canonical path correctness.

Mod 2: Deterministic collation_id stability
  - same input → same ID
  - reordered sleeve reads → same ID
  - changed weight → different ID
  - non-semantic diagnostic change → same ID

Mod 4: Allocator exactly-once proof
  - canonical path skips allocator (no double-scale)
  - combined weights match direct intent sum

Mod 5: Stale artifact rejection
  - wrong run_id → rejected
  - wrong asof_date → rejected
  - wrong schema_version → rejected
  - matching lineage → accepted

Mod 1: Inclusion semantics
  - OK sleeve inclusion field == 'included'
  - HALTED sleeve inclusion field == 'evaluated_empty'
  - DISABLED sleeve inclusion field == 'excluded_by_config'
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.orchestrator.canonical_intent_collator import (
    _compute_collation_id,
    collate_sleeve_decisions,
    intents_to_combined_weights,
    intents_to_per_sleeve_metrics,
    validate_canonical_intents_freshness,
)


def _write_decision(
    base: Path, sleeve: str, *,
    status: str = "ok",
    intents: list | None = None,
    diagnostics: dict | None = None,
):
    d = base / "sleeves" / sleeve
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "sleeve_decision.v1",
        "status": status,
        "sleeve": sleeve,
        "run_id": "run-test",
        "generated_by": "test",
        "intents": intents or [],
    }
    if diagnostics:
        payload["diagnostics"] = diagnostics
    (d / "decision.json").write_text(json.dumps(payload), encoding="utf-8")


# ── Mod 2: collation_id stability ────────────────────────────────────


class TestCollationIdStability:
    def test_same_input_same_id(self):
        intents = [
            {"sleeve": "core", "symbol": "ES", "target_weight": 0.02, "execution_phase": "futures_open", "multiplier": 50.0},
        ]
        id1 = _compute_collation_id("run-1", intents)
        id2 = _compute_collation_id("run-1", intents)
        assert id1 == id2

    def test_different_weight_different_id(self):
        i1 = [{"sleeve": "core", "symbol": "ES", "target_weight": 0.02}]
        i2 = [{"sleeve": "core", "symbol": "ES", "target_weight": 0.03}]
        assert _compute_collation_id("r", i1) != _compute_collation_id("r", i2)

    def test_reorder_stable(self):
        intents_a = [
            {"sleeve": "core", "symbol": "ES", "target_weight": 0.02},
            {"sleeve": "vrp", "symbol": "SPY", "target_weight": 0.05},
        ]
        intents_b = [
            {"sleeve": "vrp", "symbol": "SPY", "target_weight": 0.05},
            {"sleeve": "core", "symbol": "ES", "target_weight": 0.02},
        ]
        assert _compute_collation_id("r", intents_a) == _compute_collation_id("r", intents_b)

    def test_diagnostic_change_same_id(self):
        """Non-semantic fields (intent_id, generated_by) should not affect collation_id."""
        i1 = [{"sleeve": "core", "symbol": "ES", "target_weight": 0.02, "intent_id": "aaa"}]
        i2 = [{"sleeve": "core", "symbol": "ES", "target_weight": 0.02, "intent_id": "bbb"}]
        assert _compute_collation_id("r", i1) == _compute_collation_id("r", i2)

    def test_different_run_id_different_id(self):
        intents = [{"sleeve": "core", "symbol": "ES", "target_weight": 0.02}]
        assert _compute_collation_id("run-1", intents) != _compute_collation_id("run-2", intents)

    def test_collation_via_collate_stable_across_reads(self, tmp_path: Path):
        """Full round-trip: same decisions → same collation_id even if read twice."""
        _write_decision(tmp_path, "core", intents=[
            {"sleeve": "core", "symbol": "ES", "target_weight": 0.02, "execution_phase": "futures_open", "multiplier": 50.0},
        ])
        r1 = collate_sleeve_decisions(tmp_path, expected_sleeves=["core"], run_id="r1")
        r2 = collate_sleeve_decisions(tmp_path, expected_sleeves=["core"], run_id="r1")
        assert r1["collation_id"] == r2["collation_id"]


# ── Mod 1: inclusion semantics ────────────────────────────────────────


class TestInclusionSemantics:
    def test_ok_sleeve_included(self, tmp_path: Path):
        _write_decision(tmp_path, "core", intents=[
            {"sleeve": "core", "symbol": "ES", "target_weight": 0.02},
        ])
        result = collate_sleeve_decisions(tmp_path, expected_sleeves=["core"], run_id="r")
        assert result["sleeve_statuses"]["core"]["inclusion"] == "included"

    def test_halted_sleeve_evaluated_empty(self, tmp_path: Path):
        _write_decision(tmp_path, "vrp", status="halted")
        result = collate_sleeve_decisions(tmp_path, expected_sleeves=["vrp"], run_id="r")
        assert result["sleeve_statuses"]["vrp"]["inclusion"] == "evaluated_empty"
        assert result["total_intents"] == 0

    def test_disabled_sleeve_excluded_by_config(self, tmp_path: Path):
        _write_decision(tmp_path, "futures_overnight", status="disabled")
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["futures_overnight"], run_id="r",
        )
        assert result["sleeve_statuses"]["futures_overnight"]["inclusion"] == "excluded_by_config"

    def test_failed_sleeve_blocks(self, tmp_path: Path):
        _write_decision(tmp_path, "core", status="failed")
        with pytest.raises(RuntimeError, match="failed sleeves"):
            collate_sleeve_decisions(tmp_path, expected_sleeves=["core"], run_id="r")

    def test_inclusion_policy_in_result(self, tmp_path: Path):
        _write_decision(tmp_path, "core", intents=[])
        result = collate_sleeve_decisions(tmp_path, expected_sleeves=["core"], run_id="r")
        assert result["inclusion_policy"]["ok"] == "include_intents"
        assert result["inclusion_policy"]["failed"] == "hard_fail"


# ── Mod 4: allocator exactly-once ─────────────────────────────────────


class TestAllocatorExactlyOnce:
    """Prove canonical path does not double-apply allocator scaling."""

    def test_canonical_weights_are_direct_sum(self, tmp_path: Path):
        """Canonical intents already carry final weights — no allocator scaling."""
        _write_decision(tmp_path, "core", intents=[
            {"sleeve": "core", "symbol": "ES", "target_weight": 0.02},
        ])
        _write_decision(tmp_path, "vrp", intents=[
            {"sleeve": "vrp", "symbol": "SPY", "target_weight": 0.05},
            {"sleeve": "vrp", "symbol": "TLT", "target_weight": -0.03},
        ])
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["core", "vrp"], run_id="r",
        )
        combined = intents_to_combined_weights(result["intents"])
        assert abs(combined["ES"] - 0.02) < 1e-9
        assert abs(combined["SPY"] - 0.05) < 1e-9
        assert abs(combined["TLT"] - (-0.03)) < 1e-9

    def test_per_sleeve_gross_matches_intents_exactly(self, tmp_path: Path):
        _write_decision(tmp_path, "core", intents=[
            {"sleeve": "core", "symbol": "ES", "target_weight": 0.02, "intent_id": "i1"},
        ])
        _write_decision(tmp_path, "selector", intents=[
            {"sleeve": "selector", "symbol": "AAPL", "target_weight": 0.03, "intent_id": "i2"},
            {"sleeve": "selector", "symbol": "MSFT", "target_weight": -0.02, "intent_id": "i3"},
        ])
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["core", "selector"], run_id="r",
        )
        per_sleeve = intents_to_per_sleeve_metrics(result["intents"])
        assert per_sleeve["core"]["gross"] == pytest.approx(0.02)
        assert per_sleeve["selector"]["gross"] == pytest.approx(0.05)
        assert per_sleeve["selector"]["net"] == pytest.approx(0.01)

    def test_risk_and_planner_see_identical_gross(self, tmp_path: Path):
        """Prove risk gross == planner gross from same intent set."""
        _write_decision(tmp_path, "core", intents=[
            {"sleeve": "core", "symbol": "ES", "target_weight": 0.03},
        ])
        _write_decision(tmp_path, "vrp", intents=[
            {"sleeve": "vrp", "symbol": "SPY", "target_weight": 0.05},
        ])
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["core", "vrp"], run_id="r",
        )
        # Simulate risk view
        risk_combined = intents_to_combined_weights(result["intents"])
        risk_gross = sum(abs(w) for w in risk_combined.values())
        # Simulate planner view (same function, same data)
        planner_combined = intents_to_combined_weights(result["intents"])
        planner_gross = sum(abs(w) for w in planner_combined.values())
        assert risk_gross == planner_gross
        assert abs(risk_gross - 0.08) < 1e-9


# ── Mod 5: stale artifact rejection ──────────────────────────────────


class TestStaleArtifactRejection:
    def test_matching_lineage_accepted(self):
        collation = {"schema_version": "canonical_intents.v1", "run_id": "r1", "asof_date": "2026-03-07"}
        validate_canonical_intents_freshness(collation, expected_run_id="r1", expected_asof_date="2026-03-07")

    def test_wrong_run_id_rejected(self):
        collation = {"schema_version": "canonical_intents.v1", "run_id": "r1", "asof_date": "2026-03-07"}
        with pytest.raises(RuntimeError, match="stale"):
            validate_canonical_intents_freshness(collation, expected_run_id="r2", expected_asof_date="2026-03-07")

    def test_wrong_asof_date_rejected(self):
        collation = {"schema_version": "canonical_intents.v1", "run_id": "r1", "asof_date": "2026-03-06"}
        with pytest.raises(RuntimeError, match="stale"):
            validate_canonical_intents_freshness(collation, expected_run_id="r1", expected_asof_date="2026-03-07")

    def test_wrong_schema_version_rejected(self):
        collation = {"schema_version": "canonical_intents.v2", "run_id": "r1", "asof_date": "2026-03-07"}
        with pytest.raises(RuntimeError, match="schema_version"):
            validate_canonical_intents_freshness(collation, expected_run_id="r1", expected_asof_date="2026-03-07")

    def test_empty_expected_values_skips_validation(self):
        """If expected values are empty, validation is lenient (backward compat)."""
        collation = {"schema_version": "canonical_intents.v1", "run_id": "r1", "asof_date": "2026-03-07"}
        validate_canonical_intents_freshness(collation, expected_run_id="", expected_asof_date="")
