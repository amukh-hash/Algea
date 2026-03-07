from __future__ import annotations

import logging
import hashlib
from typing import Any

from .models import (
    BrokerPosition,
    ExecutionResult,
    PositionDeltaPlan,
    RiskDecisionReport,
    SleeveOutput,
)

logger = logging.getLogger(__name__)


def _compat_hit(boundary: str, message: str, **fields: Any) -> None:
    """Structured compatibility-mode telemetry event."""
    logger.warning(
        "contract_compat_hit boundary=%s message=%s fields=%s",
        boundary,
        message,
        fields,
    )


def validate_sleeve_output(payload: dict[str, Any], *, job_name: str, compatibility_mode: bool = True) -> dict[str, Any]:
    model = SleeveOutput.model_validate(payload)

    if str(model.status).lower() != "ok":
        return payload

    has_targets = isinstance(model.targets, list)
    has_intents = isinstance(model.intents, list)
    if has_targets or has_intents:
        return payload

    artifacts = model.artifacts
    if isinstance(artifacts, dict):
        keys = {str(k).lower() for k in artifacts.keys()}
        if any(k.endswith("targets") or k.endswith("intents") for k in keys):
            _compat_hit(
                "sleeve_output",
                "legacy_artifact_only_output",
                job_name=job_name,
                artifact_keys=sorted(keys),
            )
            return payload

    if compatibility_mode:
        _compat_hit(
            "sleeve_output",
            "missing_targets_or_intents_on_ok",
            job_name=job_name,
        )
        return payload

    raise ValueError(f"sleeve output invalid for {job_name}: ok status requires targets or intents")


def validate_risk_decision_report(payload: dict[str, Any], *, compatibility_mode: bool = True) -> dict[str, Any]:
    has_canonical_shape = all(
        k in payload
        for k in ("schema_version", "decision_id", "metrics", "limits", "violations")
    )
    if not has_canonical_shape and compatibility_mode:
        # Force legacy reconstruction in compatibility mode so old shapes
        # remain observable and normalized at the boundary.
        model = None
    else:
        model = True

    try:
        if model is not None:
            model_obj = RiskDecisionReport.model_validate(payload)
            if model_obj.schema_version != "risk_decision.v1":
                raise ValueError("risk decision schema_version must be risk_decision.v1")
            if not isinstance(model_obj.violations, list):
                raise ValueError("violations must be a list")
            return payload
    except Exception:
        if not compatibility_mode:
            raise

    # Legacy compatibility shim: reconstruct minimum canonical fields.
    status = str(payload.get("status", "failed"))
    reason = payload.get("reason")
    missing = payload.get("missing_sleeves", []) if isinstance(payload.get("missing_sleeves", []), list) else []
    violations: list[dict[str, Any]] = []
    if missing:
        violations.append({"code": "MISSING_TARGETS", "message": "missing targets", "details": {"missing_sleeves": missing}})
    if payload.get("nan_or_inf"):
        violations.append({"code": "NAN_INF", "message": "NaN/Inf detected", "details": {}})
    if status != "ok" and not reason:
        reason = "legacy risk report failure"

    decision_id = hashlib.sha1(
        f"{payload.get('asof_date','')}|{payload.get('session','')}|{status}|{reason}|{len(violations)}".encode("utf-8")
    ).hexdigest()[:16]

    _compat_hit(
        "risk_decision_report",
        "legacy_shape_accepted",
        status=status,
    )
    return {
        "schema_version": "risk_decision.v1",
        "status": status,
        "decision_id": decision_id,
        "policy_version": payload.get("policy_version", "risk_decision_policy.v1"),
        "input_contract_family": payload.get("input_contract_family", "legacy_read_compat"),
        "source_sleeves": payload.get("source_sleeves", []),
        "input_artifact_refs": payload.get("input_artifact_refs", {}),
        "generated_by": payload.get("generated_by", "legacy_read_compat"),
        "asof_date": payload.get("asof_date"),
        "session": payload.get("session"),
        "checked_at": payload.get("checked_at"),
        "reason": reason,
        "missing_sleeves": missing,
        "violations": violations,
        "metrics": payload.get("metrics", {}),
        "limits": payload.get("limits", {}),
        "inputs": payload.get("inputs", {}),
    }


def validate_position_delta_plan(payload: dict[str, Any], *, compatibility_mode: bool = True) -> dict[str, Any]:
    # Position plan should be canonical on default path; fail if unsafe.
    PositionDeltaPlan.model_validate(payload)
    return payload


def normalize_broker_positions_payload(payload: dict[str, Any], *, source: str, compatibility_mode: bool = True) -> dict[str, Any]:
    positions_raw = payload.get("positions", []) if isinstance(payload, dict) else []
    if not isinstance(positions_raw, list):
        if compatibility_mode:
            _compat_hit("broker_position", "positions_not_list", source=source)
            positions_raw = []
        else:
            raise ValueError("positions payload must be a list")

    normalized: list[dict[str, Any]] = []
    for row in positions_raw:
        if not isinstance(row, dict):
            if compatibility_mode:
                _compat_hit("broker_position", "position_row_not_dict", source=source)
                continue
            raise ValueError("position row must be dict")

        quantity = row.get("quantity")
        if quantity is None and "qty" in row:
            quantity = row.get("qty")
            _compat_hit("broker_position", "qty_alias_used", source=source, symbol=row.get("symbol"))

        if quantity is None:
            if compatibility_mode:
                _compat_hit("broker_position", "missing_quantity", source=source, symbol=row.get("symbol"))
                quantity = 0.0
            else:
                raise ValueError("position row missing quantity")

        canonical = {
            **row,
            "quantity": float(quantity),
            "avg_cost": float(row.get("avg_cost", row.get("avgCost", 0.0))),
            "symbol": str(row.get("symbol", "")).strip(),
        }
        broker_pos = BrokerPosition.model_validate(canonical)
        normalized.append(broker_pos.model_dump())

    out = dict(payload) if isinstance(payload, dict) else {}
    out["positions"] = normalized
    return out


def validate_execution_result(payload: dict[str, Any], *, compatibility_mode: bool = True) -> dict[str, Any]:
    routed = payload.get("routed", []) if isinstance(payload, dict) else []
    if isinstance(routed, list):
        normalized_routed: list[dict[str, Any]] = []
        for leg in routed:
            if not isinstance(leg, dict):
                if compatibility_mode:
                    _compat_hit("execution_result", "routed_leg_not_dict")
                    continue
                raise ValueError("execution leg must be dict")
            ticker = leg.get("ticker") or leg.get("symbol")
            qty = leg.get("qty", leg.get("quantity"))
            if ticker is None or qty is None:
                if compatibility_mode:
                    _compat_hit("execution_result", "missing_ticker_or_qty", leg=leg)
                    ticker = str(ticker or "")
                    qty = float(qty or 0.0)
                else:
                    raise ValueError("execution leg missing ticker/qty")
            normalized_routed.append({
                **leg,
                "ticker": str(ticker),
                "qty": float(qty),
                "side": str(leg.get("side", "")),
                "status": str(leg.get("status", "")),
            })
        payload = {**payload, "routed": normalized_routed}

    if payload.get("status") is None:
        if compatibility_mode:
            _compat_hit("execution_result", "missing_status")
            payload = {**payload, "status": "unknown"}
        else:
            raise ValueError("execution result missing status")

    if payload.get("order_count") is None:
        payload = {**payload, "order_count": len(payload.get("routed", []))}
        _compat_hit("execution_result", "missing_order_count_derived", order_count=payload["order_count"])

    ExecutionResult.model_validate(payload)
    return payload
