from __future__ import annotations

import pytest

from backend.app.contracts.validators import (
    normalize_broker_positions_payload,
    validate_execution_result,
    validate_position_delta_plan,
    validate_risk_decision_report,
    validate_sleeve_output,
)


def test_validate_sleeve_output_accepts_legacy_artifact_only_in_compat_mode():
    payload = {
        "status": "ok",
        "artifacts": {"core_targets": "backend/artifacts/x/core_targets.json"},
    }
    out = validate_sleeve_output(payload, job_name="signals_generate_core", compatibility_mode=True)
    assert out["status"] == "ok"


def test_validate_sleeve_output_fails_in_strict_mode_without_targets_or_intents():
    payload = {"status": "ok", "artifacts": {"note": "none"}}
    with pytest.raises(ValueError):
        validate_sleeve_output(payload, job_name="signals_generate_core", compatibility_mode=False)


def test_validate_risk_decision_report_accepts_legacy_shape_in_compat_mode():
    legacy = {"status": "failed", "nan_or_inf": True, "missing_sleeves": ["core"]}
    out = validate_risk_decision_report(legacy, compatibility_mode=True)
    assert out["schema_version"] == "risk_decision.v1"
    assert out["decision_id"]
    assert out["status"] == "failed"
    assert out["violations"]


def test_validate_position_delta_plan_rejects_non_positive_qty():
    payload = {
        "asof_date": "2026-02-17",
        "session": "open",
        "mode": "paper",
        "dry_run": True,
        "orders": [{"symbol": "SPY", "qty": 0, "side": "BUY"}],
        "summary": {},
    }
    with pytest.raises(Exception):
        validate_position_delta_plan(payload, compatibility_mode=True)


def test_normalize_broker_positions_payload_accepts_qty_alias():
    payload = {"positions": [{"symbol": "SPY", "qty": 3, "avg_cost": 500.0}]}
    out = normalize_broker_positions_payload(payload, source="test", compatibility_mode=True)
    assert out["positions"][0]["quantity"] == 3.0


def test_validate_execution_result_accepts_compat_symbol_quantity_fields():
    payload = {
        "status": "accepted",
        "routed": [{"symbol": "SPY", "quantity": 2, "side": "BUY", "status": "filled"}],
    }
    out = validate_execution_result(payload, compatibility_mode=True)
    assert out["order_count"] == 1
    assert out["routed"][0]["ticker"] == "SPY"
    assert out["routed"][0]["qty"] == 2.0
