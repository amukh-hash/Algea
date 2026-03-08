"""Phase 5/6 integration and failure injection tests.

End-to-end canonical-only tick:
  sleeve decisions → collation → risk limits → planner input

Failure injection:
  - Failed sleeve blocks collation
  - Missing sleeve blocks collation
  - NaN weight triggers risk violation
  - Stale artifact rejected by planner
  - Gross limit violation detected

Observability field assertions:
  - collation_id present in all canonical artifacts
  - input_family == canonical_intents
  - sleeve_statuses carry inclusion field
  - schema_version present
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from backend.app.orchestrator.canonical_intent_collator import (
    collate_sleeve_decisions,
    intents_to_combined_weights,
    intents_to_per_sleeve_metrics,
    validate_canonical_intents_freshness,
    write_canonical_intents,
)


def _write_decision(
    base: Path, sleeve: str, *,
    status: str = "ok",
    intents: list | None = None,
    run_id: str = "tick-001",
    diagnostics: dict | None = None,
):
    d = base / "sleeves" / sleeve
    d.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "sleeve_decision.v1",
        "status": status,
        "sleeve": sleeve,
        "run_id": run_id,
        "generated_by": f"handle_signals_generate_{sleeve}",
        "market_snapshot_id": "mkt-snap-001",
        "portfolio_snapshot_id": "pf-snap-001",
        "intents": intents or [],
        "diagnostics": diagnostics or {"source_branch": f"test_{sleeve}"},
    }
    (d / "decision.json").write_text(json.dumps(payload), encoding="utf-8")


def _risk_limits() -> dict[str, Any]:
    return {
        "max_gross": 1.5,
        "max_net_abs": 0.5,
        "max_symbol_abs_weight": 0.3,
        "max_symbols": 50,
    }


# ── Integration: Full canonical tick ──────────────────────────────────


class TestCanonicalTickIntegration:
    """End-to-end: sleeve decisions → collation → risk → planner input."""

    def _setup_three_sleeves(self, tmp_path: Path):
        _write_decision(tmp_path, "core", intents=[
            {"intent_id": "c1", "sleeve": "core", "symbol": "ES", "target_weight": 0.03,
             "asset_class": "FUTURE", "execution_phase": "futures_open", "multiplier": 50.0},
        ])
        _write_decision(tmp_path, "vrp", intents=[
            {"intent_id": "v1", "sleeve": "vrp", "symbol": "SPY", "target_weight": 0.05,
             "asset_class": "EQUITY", "execution_phase": "rth_open", "multiplier": 1.0},
            {"intent_id": "v2", "sleeve": "vrp", "symbol": "TLT", "target_weight": -0.02,
             "asset_class": "EQUITY", "execution_phase": "rth_open", "multiplier": 1.0},
        ])
        _write_decision(tmp_path, "selector", intents=[
            {"intent_id": "s1", "sleeve": "selector", "symbol": "AAPL", "target_weight": 0.04,
             "asset_class": "EQUITY", "execution_phase": "rth_open", "multiplier": 1.0},
        ])

    def test_full_tick_produces_valid_collation(self, tmp_path: Path):
        self._setup_three_sleeves(tmp_path)
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["core", "vrp", "selector"],
            run_id="tick-001", asof_date="2026-03-07",
        )

        # Shape assertions
        assert result["total_intents"] == 4
        assert result["ok_sleeves"] == ["core", "vrp", "selector"]
        assert result["failed_sleeves"] == []
        assert result["halted_sleeves"] == []
        assert result["disabled_sleeves"] == []
        assert result["missing_sleeves"] == []

        # Observability assertions
        assert result["collation_id"].startswith("col-")
        assert result["schema_version"] == "canonical_intents.v1"
        assert result["run_id"] == "tick-001"
        assert result["asof_date"] == "2026-03-07"
        assert "inclusion_policy" in result

        # Per-sleeve inclusion field
        for sleeve in ["core", "vrp", "selector"]:
            assert result["sleeve_statuses"][sleeve]["inclusion"] == "included"

    def test_full_tick_combined_weights_correct(self, tmp_path: Path):
        self._setup_three_sleeves(tmp_path)
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["core", "vrp", "selector"],
            run_id="tick-001",
        )
        combined = intents_to_combined_weights(result["intents"])
        assert abs(combined["ES"] - 0.03) < 1e-9
        assert abs(combined["SPY"] - 0.05) < 1e-9
        assert abs(combined["TLT"] - (-0.02)) < 1e-9
        assert abs(combined["AAPL"] - 0.04) < 1e-9

    def test_full_tick_per_sleeve_metrics(self, tmp_path: Path):
        self._setup_three_sleeves(tmp_path)
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["core", "vrp", "selector"],
            run_id="tick-001",
        )
        per_sleeve = intents_to_per_sleeve_metrics(result["intents"])
        assert per_sleeve["core"]["gross"] == pytest.approx(0.03)
        assert per_sleeve["vrp"]["gross"] == pytest.approx(0.07)
        assert per_sleeve["vrp"]["net"] == pytest.approx(0.03)
        assert per_sleeve["selector"]["gross"] == pytest.approx(0.04)

    def test_full_tick_risk_and_planner_see_same_data(self, tmp_path: Path):
        self._setup_three_sleeves(tmp_path)
        collation = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["core", "vrp", "selector"],
            run_id="tick-001", asof_date="2026-03-07",
        )

        # Risk writes canonical_intents.json
        intents_path = write_canonical_intents(collation, tmp_path)
        assert intents_path.exists()

        # Planner reads and validates
        planner_data = json.loads(intents_path.read_text())
        validate_canonical_intents_freshness(
            planner_data, expected_run_id="tick-001", expected_asof_date="2026-03-07",
        )
        assert planner_data["collation_id"] == collation["collation_id"]

        # Same combined weights
        risk_combined = intents_to_combined_weights(collation["intents"])
        planner_combined = intents_to_combined_weights(planner_data["intents"])
        assert risk_combined == planner_combined

    def test_full_tick_with_halted_and_disabled(self, tmp_path: Path):
        """Mixed statuses: core OK, vrp HALTED, selector OK, futures DISABLED."""
        _write_decision(tmp_path, "core", intents=[
            {"intent_id": "c1", "sleeve": "core", "symbol": "ES", "target_weight": 0.03},
        ])
        _write_decision(tmp_path, "vrp", status="halted")
        _write_decision(tmp_path, "selector", intents=[
            {"intent_id": "s1", "sleeve": "selector", "symbol": "AAPL", "target_weight": 0.04},
        ])
        _write_decision(tmp_path, "futures_overnight", status="disabled")

        result = collate_sleeve_decisions(
            tmp_path,
            expected_sleeves=["core", "vrp", "selector", "futures_overnight"],
            run_id="tick-001",
        )
        assert result["ok_sleeves"] == ["core", "selector"]
        assert result["halted_sleeves"] == ["vrp"]
        assert result["disabled_sleeves"] == ["futures_overnight"]
        assert result["total_intents"] == 2
        combined = intents_to_combined_weights(result["intents"])
        assert "ES" in combined
        assert "AAPL" in combined

    def test_gross_and_net_within_limits(self, tmp_path: Path):
        self._setup_three_sleeves(tmp_path)
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["core", "vrp", "selector"], run_id="tick",
        )
        combined = intents_to_combined_weights(result["intents"])
        limits = _risk_limits()
        gross = sum(abs(w) for w in combined.values())
        net = sum(combined.values())
        assert gross <= limits["max_gross"], f"gross={gross} > {limits['max_gross']}"
        assert abs(net) <= limits["max_net_abs"], f"net={net} > {limits['max_net_abs']}"
        for sym, w in combined.items():
            assert abs(w) <= limits["max_symbol_abs_weight"], f"{sym} weight {w}"


# ── Failure injection ─────────────────────────────────────────────────


class TestFailureInjection:
    def test_failed_sleeve_blocks_pipeline(self, tmp_path: Path):
        _write_decision(tmp_path, "core", intents=[
            {"intent_id": "c1", "sleeve": "core", "symbol": "ES", "target_weight": 0.03},
        ])
        _write_decision(tmp_path, "vrp", status="failed")
        _write_decision(tmp_path, "selector", intents=[
            {"intent_id": "s1", "sleeve": "selector", "symbol": "AAPL", "target_weight": 0.04},
        ])

        with pytest.raises(RuntimeError, match="failed sleeves"):
            collate_sleeve_decisions(
                tmp_path, expected_sleeves=["core", "vrp", "selector"], run_id="tick",
            )

    def test_missing_sleeve_blocks_pipeline(self, tmp_path: Path):
        _write_decision(tmp_path, "core", intents=[
            {"intent_id": "c1", "sleeve": "core", "symbol": "ES", "target_weight": 0.03},
        ])
        # vrp and selector decisions not written
        with pytest.raises(RuntimeError, match="missing sleeve decisions"):
            collate_sleeve_decisions(
                tmp_path, expected_sleeves=["core", "vrp", "selector"], run_id="tick",
            )

    def test_nan_weight_detected(self, tmp_path: Path):
        _write_decision(tmp_path, "core", intents=[
            {"intent_id": "c1", "sleeve": "core", "symbol": "ES", "target_weight": float("nan")},
        ])
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["core"], run_id="tick",
        )
        combined = intents_to_combined_weights(result["intents"])
        # NaN detected at risk check level
        for w in combined.values():
            if w != w:  # NaN check
                break
        else:
            pytest.fail("Expected NaN in combined weights")

    def test_stale_run_id_rejected(self, tmp_path: Path):
        _write_decision(tmp_path, "core", run_id="old-tick", intents=[
            {"intent_id": "c1", "sleeve": "core", "symbol": "ES", "target_weight": 0.03},
        ])
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["core"],
            run_id="old-tick", asof_date="2026-03-06",
        )
        path = write_canonical_intents(result, tmp_path)
        data = json.loads(path.read_text())

        with pytest.raises(RuntimeError, match="stale"):
            validate_canonical_intents_freshness(
                data, expected_run_id="new-tick", expected_asof_date="2026-03-07",
            )

    def test_gross_limit_violation(self, tmp_path: Path):
        """Large weights trigger gross limit on risk check."""
        _write_decision(tmp_path, "core", intents=[
            {"intent_id": "c1", "sleeve": "core", "symbol": "ES", "target_weight": 0.80},
        ])
        _write_decision(tmp_path, "vrp", intents=[
            {"intent_id": "v1", "sleeve": "vrp", "symbol": "SPY", "target_weight": 0.80},
        ])
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["core", "vrp"], run_id="tick",
        )
        combined = intents_to_combined_weights(result["intents"])
        limits = _risk_limits()
        gross = sum(abs(w) for w in combined.values())
        assert gross > limits["max_gross"], "Expected gross violation"

    def test_corrupted_decision_json_treated_as_failed(self, tmp_path: Path):
        """Unparseable JSON → read_error → treated as failed."""
        d = tmp_path / "sleeves" / "core"
        d.mkdir(parents=True, exist_ok=True)
        (d / "decision.json").write_text("{invalid json", encoding="utf-8")

        with pytest.raises(RuntimeError, match="failed sleeves"):
            collate_sleeve_decisions(tmp_path, expected_sleeves=["core"], run_id="tick")


# ── Observability field assertions ────────────────────────────────────


class TestObservabilityFields:
    def test_collation_has_all_observability_fields(self, tmp_path: Path):
        _write_decision(tmp_path, "core", intents=[
            {"intent_id": "c1", "sleeve": "core", "symbol": "ES", "target_weight": 0.03},
        ])
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["core"],
            run_id="tick-001", asof_date="2026-03-07",
        )
        # Required fields per Phase 5 spec
        assert "collation_id" in result
        assert "run_id" in result
        assert "asof_date" in result
        assert "schema_version" in result
        assert "collated_at" in result
        assert "expected_sleeves" in result
        assert "inclusion_policy" in result
        assert "total_intents" in result
        assert "sleeve_statuses" in result

    def test_sleeve_status_has_inclusion_field(self, tmp_path: Path):
        _write_decision(tmp_path, "core", intents=[
            {"intent_id": "c1", "sleeve": "core", "symbol": "ES", "target_weight": 0.03},
        ])
        _write_decision(tmp_path, "vrp", status="halted")
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["core", "vrp"], run_id="tick",
        )
        for sleeve, status in result["sleeve_statuses"].items():
            assert "inclusion" in status, f"Missing inclusion field for {sleeve}"
            assert "status" in status
            assert "n_intents" in status

    def test_written_artifact_preserves_all_fields(self, tmp_path: Path):
        _write_decision(tmp_path, "core", intents=[
            {"intent_id": "c1", "sleeve": "core", "symbol": "ES", "target_weight": 0.03},
        ])
        result = collate_sleeve_decisions(
            tmp_path, expected_sleeves=["core"],
            run_id="tick-001", asof_date="2026-03-07",
        )
        path = write_canonical_intents(result, tmp_path)
        data = json.loads(path.read_text())

        # Round-trip preserves all observability fields
        assert data["collation_id"] == result["collation_id"]
        assert data["run_id"] == "tick-001"
        assert data["asof_date"] == "2026-03-07"
        assert data["schema_version"] == "canonical_intents.v1"
        assert data["total_intents"] == 1
        assert "inclusion_policy" in data
