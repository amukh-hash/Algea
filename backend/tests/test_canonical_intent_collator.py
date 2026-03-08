"""Phase 4 tests for canonical_intent_collator.

Validates:
  1. collate_sleeve_decisions reads decision.json files correctly
  2. Deterministic collation_id for risk==planner invariant
  3. Failed sleeves block collation
  4. Missing sleeves block collation when fail_on_missing=True
  5. HALTED/DISABLED sleeves produce 0 intents but don't block
  6. intents_to_combined_weights aggregates per-symbol weights
  7. intents_to_per_sleeve_metrics computes per-sleeve gross/net
  8. write_canonical_intents writes JSON file
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.orchestrator.canonical_intent_collator import (
    collate_sleeve_decisions,
    intents_to_combined_weights,
    intents_to_per_sleeve_metrics,
    write_canonical_intents,
)


def _write_decision(base: Path, sleeve: str, *, status: str = "ok", intents: list | None = None):
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
    (d / "decision.json").write_text(json.dumps(payload), encoding="utf-8")


class TestCollateSleeveDecisions:
    def test_ok_sleeves_collated(self, tmp_path: Path):
        _write_decision(tmp_path, "core", intents=[
            {"intent_id": "i1", "symbol": "ES", "target_weight": 0.02, "sleeve": "core"},
        ])
        _write_decision(tmp_path, "vrp", intents=[
            {"intent_id": "i2", "symbol": "SPY", "target_weight": 0.05, "sleeve": "vrp"},
        ])
        _write_decision(tmp_path, "selector", intents=[
            {"intent_id": "i3", "symbol": "AAPL", "target_weight": 0.03, "sleeve": "selector"},
        ])

        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["core", "vrp", "selector"], run_id="r1",
        )
        assert result["total_intents"] == 3
        assert result["ok_sleeves"] == ["core", "vrp", "selector"]
        assert result["failed_sleeves"] == []
        assert result["collation_id"].startswith("col-")

    def test_deterministic_collation_id(self, tmp_path: Path):
        _write_decision(tmp_path, "core", intents=[
            {"intent_id": "i1", "symbol": "ES", "target_weight": 0.02, "sleeve": "core"},
        ])
        r1 = collate_sleeve_decisions(tmp_path, expected_sleeves=["core"], run_id="r1")
        r2 = collate_sleeve_decisions(tmp_path, expected_sleeves=["core"], run_id="r1")
        assert r1["collation_id"] == r2["collation_id"]

    def test_failed_sleeve_blocks(self, tmp_path: Path):
        _write_decision(tmp_path, "core", status="failed")
        with pytest.raises(RuntimeError, match="failed sleeves"):
            collate_sleeve_decisions(tmp_path, expected_sleeves=["core"], run_id="r1")

    def test_missing_sleeve_blocks(self, tmp_path: Path):
        with pytest.raises(RuntimeError, match="missing sleeve decisions"):
            collate_sleeve_decisions(
                tmp_path, expected_sleeves=["core"], run_id="r1", fail_on_missing=True,
            )

    def test_missing_sleeve_allowed(self, tmp_path: Path):
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["core"], run_id="r1", fail_on_missing=False,
        )
        assert result["missing_sleeves"] == ["core"]
        assert result["total_intents"] == 0

    def test_halted_sleeve_no_intents(self, tmp_path: Path):
        _write_decision(tmp_path, "vrp", status="halted")
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["vrp"], run_id="r1",
        )
        assert result["halted_sleeves"] == ["vrp"]
        assert result["total_intents"] == 0

    def test_disabled_sleeve_no_intents(self, tmp_path: Path):
        _write_decision(tmp_path, "futures_overnight", status="disabled")
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["futures_overnight"], run_id="r1",
        )
        assert result["disabled_sleeves"] == ["futures_overnight"]
        assert result["total_intents"] == 0

    def test_mixed_statuses(self, tmp_path: Path):
        _write_decision(tmp_path, "core", intents=[
            {"intent_id": "i1", "symbol": "ES", "target_weight": 0.02, "sleeve": "core"},
        ])
        _write_decision(tmp_path, "vrp", status="halted")
        _write_decision(tmp_path, "futures_overnight", status="disabled")
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["core", "vrp", "futures_overnight"], run_id="r1",
        )
        assert result["ok_sleeves"] == ["core"]
        assert result["halted_sleeves"] == ["vrp"]
        assert result["disabled_sleeves"] == ["futures_overnight"]
        assert result["total_intents"] == 1


class TestIntentsToCombinedWeights:
    def test_aggregates(self):
        intents = [
            {"symbol": "ES", "target_weight": 0.02},
            {"symbol": "ES", "target_weight": 0.01},
            {"symbol": "SPY", "target_weight": 0.05},
        ]
        combined = intents_to_combined_weights(intents)
        assert abs(combined["ES"] - 0.03) < 1e-9
        assert abs(combined["SPY"] - 0.05) < 1e-9

    def test_skips_empty_symbol(self):
        intents = [{"symbol": "", "target_weight": 0.05}]
        combined = intents_to_combined_weights(intents)
        assert combined == {}


class TestIntentsToPerSleeveMetrics:
    def test_computes_per_sleeve(self):
        intents = [
            {"symbol": "ES", "target_weight": 0.02, "sleeve": "core", "intent_id": "i1"},
            {"symbol": "SPY", "target_weight": -0.03, "sleeve": "vrp", "intent_id": "i2"},
            {"symbol": "AAPL", "target_weight": 0.01, "sleeve": "vrp", "intent_id": "i3"},
        ]
        per_sleeve = intents_to_per_sleeve_metrics(intents)
        assert per_sleeve["core"]["gross"] == pytest.approx(0.02)
        assert per_sleeve["vrp"]["gross"] == pytest.approx(0.04)
        assert per_sleeve["vrp"]["net"] == pytest.approx(-0.02)
        assert per_sleeve["vrp"]["num_symbols"] == 2


class TestWriteCanonicalIntents:
    def test_writes_json(self, tmp_path: Path):
        collation = {
            "collation_id": "col-test",
            "intents": [{"intent_id": "i1", "symbol": "ES"}],
        }
        path = write_canonical_intents(collation, tmp_path)
        assert path.exists()
        data = json.loads(path.read_text())
        assert data["collation_id"] == "col-test"
        assert len(data["intents"]) == 1
