from __future__ import annotations

import json
from pathlib import Path

import pytest

from backend.app.core.schemas import ExecutionPhase
from backend.app.orchestrator.intent_translation import translate_targets_to_intents
from backend.app.orchestrator.job_defs import handle_order_build_and_route
from backend.tests.test_orchestrator import SpyBroker


def _write(path: Path, payload: dict | list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_core_native_intents_are_trace_compatible(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _write(root / "targets" / "core_targets.json", {"targets": [{"symbol": "SPY", "target_weight": 0.1}]})
    _write(root / "intents" / "core_intents.json", [{
        "asof_date": "2026-02-17",
        "sleeve": "core",
        "symbol": "SPY",
        "asset_class": "EQUITY",
        "target_weight": 0.1,
        "execution_phase": "intraday",
    }])

    out = translate_targets_to_intents(
        artifact_root=root,
        asof_date="2026-02-17",
        target_paths={"core": root / "targets" / "core_targets.json"},
        allocator_scales={"core": 1.0},
    )
    payload = json.loads(Path(out.translated_intents_path).read_text(encoding="utf-8"))
    row = payload["intents"][0]
    assert row["trace"]["source_kind"] == "native_intent"
    assert row["trace"]["source_sleeve"] == "core"


def test_vrp_translation_contract(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _write(root / "targets" / "vrp_targets.json", {"targets": [{"symbol": "SPY", "target_weight": 0.2}]})

    out = translate_targets_to_intents(
        artifact_root=root,
        asof_date="2026-02-17",
        target_paths={"vrp": root / "targets" / "vrp_targets.json"},
        allocator_scales={"vrp": 0.5},
    )
    payload = json.loads(Path(out.translated_intents_path).read_text(encoding="utf-8"))
    intent = payload["intents"][0]["intent"]
    assert intent["sleeve"] == "vrp"
    assert intent["asset_class"] == "EQUITY"
    assert intent["execution_phase"] == ExecutionPhase.INTRADAY.value
    assert intent["target_weight"] == 0.1


def test_selector_parity_reports_native_intents_present(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _write(root / "targets" / "selector_targets.json", {"targets": [{"symbol": "AAPL", "target_weight": 0.1}]})
    _write(root / "intents" / "selector_intents.json", [{
        "asof_date": "2026-02-17",
        "sleeve": "selector",
        "symbol": "AAPL",
        "asset_class": "EQUITY",
        "target_weight": 0.1,
        "execution_phase": "intraday",
    }])

    out = translate_targets_to_intents(
        artifact_root=root,
        asof_date="2026-02-17",
        target_paths={"selector": root / "targets" / "selector_targets.json"},
        allocator_scales={"selector": 1.0},
    )
    parity = json.loads(Path(out.parity_report_path).read_text(encoding="utf-8"))
    assert parity["native_intents_present"]["selector"] is True


def test_futures_multiplier_lookup_success(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _write(root / "targets" / "futures_overnight_targets.json", {"targets": [{"symbol": "ES", "target_weight": 0.05}]})

    out = translate_targets_to_intents(
        artifact_root=root,
        asof_date="2026-02-17",
        target_paths={"futures_overnight": root / "targets" / "futures_overnight_targets.json"},
        allocator_scales={"futures_overnight": 1.0},
    )
    payload = json.loads(Path(out.translated_intents_path).read_text(encoding="utf-8"))
    assert payload["intents"][0]["intent"]["multiplier"] == 50.0


def test_futures_multiplier_lookup_failure(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _write(root / "targets" / "futures_overnight_targets.json", {"targets": [{"symbol": "UNKNOWN", "target_weight": 0.05}]})

    with pytest.raises(RuntimeError, match="UNTRANSLATABLE_TARGET"):
        translate_targets_to_intents(
            artifact_root=root,
            asof_date="2026-02-17",
            target_paths={"futures_overnight": root / "targets" / "futures_overnight_targets.json"},
            allocator_scales={"futures_overnight": 1.0},
        )


def test_statarb_pair_only_rejection(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _write(root / "targets" / "statarb_targets.json", {"targets": [{"pair": "X_Y", "z_score": 1.2}]})

    with pytest.raises(RuntimeError, match="UNTRANSLATABLE_TARGET"):
        translate_targets_to_intents(
            artifact_root=root,
            asof_date="2026-02-17",
            target_paths={"statarb": root / "targets" / "statarb_targets.json"},
            allocator_scales={"statarb": 1.0},
        )


def test_allocator_applied_exactly_once(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _write(root / "targets" / "vrp_targets.json", {"targets": [{"symbol": "SPY", "target_weight": 0.2}]})

    out = translate_targets_to_intents(
        artifact_root=root,
        asof_date="2026-02-17",
        target_paths={"vrp": root / "targets" / "vrp_targets.json"},
        allocator_scales={"vrp": 0.5},
    )
    payload = json.loads(Path(out.translated_intents_path).read_text(encoding="utf-8"))
    assert payload["intents"][0]["intent"]["target_weight"] == 0.1


def test_translated_intent_trace_fields_present(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _write(root / "targets" / "selector_targets.json", {"targets": [{"symbol": "MSFT", "target_weight": 0.04}]})

    out = translate_targets_to_intents(
        artifact_root=root,
        asof_date="2026-02-17",
        target_paths={"selector": root / "targets" / "selector_targets.json"},
        allocator_scales={"selector": 1.0},
    )
    row = json.loads(Path(out.translated_intents_path).read_text(encoding="utf-8"))["intents"][0]
    for key in [
        "source_sleeve",
        "source_artifact_path",
        "source_row_index",
        "policy_version",
        "allocator_scale_applied",
        "translation_timestamp",
    ]:
        assert key in row["trace"]


def test_parity_artifact_generation(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _write(root / "targets" / "core_targets.json", {"targets": [{"symbol": "SPY", "target_weight": 0.1}]})

    out = translate_targets_to_intents(
        artifact_root=root,
        asof_date="2026-02-17",
        target_paths={"core": root / "targets" / "core_targets.json"},
        allocator_scales={"core": 1.0},
    )
    parity = json.loads(Path(out.parity_report_path).read_text(encoding="utf-8"))
    assert "sleeves_processed" in parity
    assert "translated_intent_counts" in parity
    assert "source_target_counts" in parity


def test_target_driven_default_execution_path_unchanged(tmp_path: Path):
    root = tmp_path / "artifacts" / "2026-02-17"
    (root / "targets").mkdir(parents=True, exist_ok=True)
    (root / "signals").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "intents").mkdir(parents=True, exist_ok=True)

    for sleeve in ["core", "vrp", "selector", "futures_overnight", "statarb"]:
        _write(root / "targets" / f"{sleeve}_targets.json", {
            "schema_version": "targets.v1",
            "status": "ok",
            "is_stub": False,
            "targets": [{"symbol": "SPY", "target_weight": 0.02}],
        })
        _write(root / "signals" / f"{sleeve}_signals.json", {
            "schema_version": "signals.v1",
            "status": "ok",
            "is_stub": False,
        })

    # Opposite native intent should not drive planner in PR-3 (no execution cutover).
    _write(root / "intents" / "core_intents.json", [{
        "asof_date": "2026-02-17",
        "sleeve": "core",
        "symbol": "SPY",
        "asset_class": "EQUITY",
        "target_weight": -0.9,
        "execution_phase": "intraday",
    }])

    _write(root / "reports" / "risk_checks.json", {
        "status": "ok",
        "checked_at": "2026-02-17T13:00:00+00:00",
        "asof_date": "2026-02-17",
        "session": "open",
        "inputs": {"target_paths": {}},
        "metrics": {"nan_or_inf": False, "gross_exposure": 0.10, "net_exposure": 0.10, "num_symbols": 1, "per_sleeve": {}},
        "limits": {},
        "violations": [],
    })

    result = handle_order_build_and_route({
        "asof_date": "2026-02-17",
        "session": "open",
        "artifact_root": str(root),
        "mode": "paper",
        "dry_run": True,
        "broker": SpyBroker(price_map={"SPY": 500.0}),
        "config": {"account_equity": 100_000, "order_notional_rounding": 1},
    })
    assert result["status"] == "ok"
    orders = json.loads((root / "orders" / "orders.json").read_text(encoding="utf-8"))
    assert orders["orders"][0]["side"] == "BUY"
    assert (root / "intents" / "translated_from_targets_intents.json").exists()
    assert (root / "reports" / "intent_translation_parity.json").exists()
