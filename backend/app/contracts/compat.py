"""Compatibility bridge: canonical → legacy artifact adapters.

During migration, legacy ``targets/*.json`` and ``signals/*.json`` files
are *derived* from canonical ``SleeveDecision`` objects.  They carry
``authoritative: false`` to signal they are mirrors, not primary inputs.

Canonical dataclass contracts are the **authoritative internal runtime
contracts**.  The Pydantic ``TargetIntent`` in ``backend.app.core.schemas``
is an **input-compatibility parser only**.

All derived artifacts MUST include:
  * ``authoritative: false``
  * ``derived_from: "canonical_intents"``
  * ``source_run_id``  — linking back to the originating run
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from .canonical import (
    AssetClass,
    ExecutionPhase,
    SleeveDecision,
    SleeveStatus,
    TargetIntent as CanonicalTargetIntent,
    TraceRef,
)
from .errors import ContractViolationError


def sleeve_decision_to_targets_compat(decision: SleeveDecision) -> dict[str, Any]:
    """Derive a legacy ``targets.v1`` artifact from a canonical ``SleeveDecision``.

    The output always includes ``authoritative: false`` and ``source_run_id``.
    """
    targets = []
    for intent in decision.intents:
        targets.append({
            "symbol": intent.symbol,
            "target_weight": intent.target_weight,
            "asset_class": intent.asset_class.value,
            "execution_phase": intent.execution_phase.value,
            "multiplier": intent.multiplier,
            "intent_id": intent.intent_id,
        })

    return {
        "schema_version": "targets_compat.v1",
        "derived_from": "canonical_intents",
        "derived_from_schema": decision.schema_version,
        "derived_from_intent_ids": [i.intent_id for i in decision.intents],
        "authoritative": False,
        "source_run_id": decision.run_id,
        "source_sleeve_decision": f"{decision.sleeve}/sleeve_decision.v1",
        "control_snapshot_id": decision.control_snapshot_id,
        "market_snapshot_id": decision.market_snapshot_id,
        "portfolio_snapshot_id": decision.portfolio_snapshot_id,
        "status": decision.status.value,
        "sleeve": decision.sleeve,
        "asof_date": decision.asof_date.isoformat(),
        "targets": targets,
        "is_stub": False,
        "generated_at": decision.completed_at.isoformat(),
    }


def sleeve_decision_to_signals_compat(decision: SleeveDecision) -> dict[str, Any]:
    """Derive a legacy ``signals.v1`` artifact from a canonical ``SleeveDecision``.

    The output always includes ``authoritative: false`` and ``source_run_id``.
    """
    signals = []
    for intent in decision.intents:
        signals.append({
            "symbol": intent.symbol,
            "score": intent.target_weight,
            "intent_id": intent.intent_id,
        })

    return {
        "schema_version": "signals_compat.v1",
        "derived_from": "canonical_intents",
        "derived_from_schema": decision.schema_version,
        "derived_from_intent_ids": [i.intent_id for i in decision.intents],
        "authoritative": False,
        "source_run_id": decision.run_id,
        "control_snapshot_id": decision.control_snapshot_id,
        "market_snapshot_id": decision.market_snapshot_id,
        "portfolio_snapshot_id": decision.portfolio_snapshot_id,
        "status": decision.status.value,
        "sleeve": decision.sleeve,
        "asof_date": decision.asof_date.isoformat(),
        "source": "canonical_sleeve_decision",
        "signals": signals,
        "is_stub": False,
        "generated_at": decision.completed_at.isoformat(),
    }


def intents_to_targets_compat(
    intents: list[CanonicalTargetIntent] | tuple[CanonicalTargetIntent, ...],
    *,
    run_id: str = "",
) -> dict[str, Any]:
    """Derive a combined legacy targets dict from a list of canonical intents.

    The output always includes ``authoritative: false``.
    """
    by_sleeve: dict[str, list[dict[str, Any]]] = {}
    for intent in intents:
        targets = by_sleeve.setdefault(intent.sleeve, [])
        targets.append({
            "symbol": intent.symbol,
            "target_weight": intent.target_weight,
            "asset_class": intent.asset_class.value,
            "execution_phase": intent.execution_phase.value,
            "multiplier": intent.multiplier,
            "intent_id": intent.intent_id,
        })

    return {
        "schema_version": "targets_compat.v1",
        "derived_from": "canonical_intents",
        "authoritative": False,
        "source_run_id": run_id,
        "by_sleeve": by_sleeve,
        "total_intents": len(intents),
    }


def pydantic_intent_to_canonical(
    pydantic_ti: Any,
    *,
    intent_id: str | None = None,
    run_id: str = "",
    trace: TraceRef | None = None,
) -> CanonicalTargetIntent:
    """Bridge existing Pydantic ``TargetIntent`` to canonical frozen ``TargetIntent``.

    Parameters
    ----------
    pydantic_ti
        Instance of ``backend.app.core.schemas.TargetIntent`` (Pydantic model).
    intent_id
        Optional explicit ID; auto-generated if None.
    run_id
        Run/tick identifier for lineage.
    trace
        Optional full trace reference; a stub trace is created if None.

    Raises
    ------
    ContractViolationError
        If the Pydantic model contains values that cannot be safely mapped
        to canonical types (invalid enum, invalid multiplier, empty symbol).
    """
    from datetime import date as _date

    # ── Parse asof_date ──
    asof = pydantic_ti.asof_date
    if isinstance(asof, str):
        asof = _date.fromisoformat(asof)

    # ── Parse asset_class ──
    raw_ac = str(pydantic_ti.asset_class).upper()
    try:
        asset_class = AssetClass(raw_ac)
    except ValueError:
        raise ContractViolationError(
            f"Cannot map asset_class '{raw_ac}' to canonical AssetClass"
        )

    # ── Parse execution_phase ──
    phase_val = pydantic_ti.execution_phase
    if hasattr(phase_val, "value"):
        phase_val = phase_val.value
    try:
        execution_phase = ExecutionPhase(str(phase_val))
    except ValueError:
        raise ContractViolationError(
            f"Cannot map execution_phase '{phase_val}' to canonical ExecutionPhase"
        )

    # ── Validate symbol ──
    symbol = str(pydantic_ti.symbol).strip()
    if not symbol:
        raise ContractViolationError("Cannot convert intent with empty symbol")

    # ── Validate multiplier ──
    mult = float(pydantic_ti.multiplier)
    if mult <= 0:
        raise ContractViolationError(
            f"Cannot convert intent with non-positive multiplier: {mult}"
        )

    # ── Validate weight ──
    weight = float(pydantic_ti.target_weight)
    if not (-2.0 <= weight <= 2.0):
        raise ContractViolationError(
            f"Cannot convert intent with target_weight {weight} outside [-2.0, 2.0]"
        )

    if trace is None:
        trace = TraceRef(
            run_id=run_id,
            sleeve=str(pydantic_ti.sleeve),
            sleeve_run_id="",
            source_artifact="pydantic_bridge",
            source_row_index=None,
            control_snapshot_id="",
            market_snapshot_id="",
            portfolio_snapshot_id="",
        )

    return CanonicalTargetIntent(
        intent_id=intent_id or str(uuid.uuid4()),
        run_id=run_id,
        asof_date=asof,
        sleeve=str(pydantic_ti.sleeve),
        symbol=symbol,
        asset_class=asset_class,
        target_weight=weight,
        execution_phase=execution_phase,
        multiplier=mult,
        trace=trace,
        metadata={"dte": int(pydantic_ti.dte)} if hasattr(pydantic_ti, "dte") else {},
    )
