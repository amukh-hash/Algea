"""Shared helpers for emitting canonical SleeveDecision from sleeve handlers.

Used by Phase 3 remediated handlers when ``FF_CANONICAL_SLEEVE_OUTPUTS`` is on.
Writes decision artifacts to ``sleeves/<name>/decision.json`` and
compatibility artifacts to ``targets/`` and ``signals/`` with
``authoritative: false``.
"""
from __future__ import annotations

import json
import logging
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.contracts.canonical import (
    AssetClass,
    ExecutionPhase,
    SleeveDecision,
    SleeveStatus,
    TargetIntent,
    TraceRef,
)
from backend.app.contracts.compat import (
    sleeve_decision_to_signals_compat,
    sleeve_decision_to_targets_compat,
)
from backend.app.version import with_app_metadata

logger = logging.getLogger(__name__)


def build_trace(
    *,
    run_id: str,
    sleeve: str,
    control_snapshot_id: str = "",
    market_snapshot_id: str = "",
    portfolio_snapshot_id: str = "",
    source_artifact: str = "handler",
    model_version: str | None = None,
    config_version: str | None = None,
) -> TraceRef:
    """Build a standard TraceRef for a sleeve handler."""
    return TraceRef(
        run_id=run_id,
        sleeve=sleeve,
        sleeve_run_id=f"{sleeve}-{run_id}",
        source_artifact=source_artifact,
        source_row_index=None,
        control_snapshot_id=control_snapshot_id,
        market_snapshot_id=market_snapshot_id,
        portfolio_snapshot_id=portfolio_snapshot_id,
        model_version=model_version,
        config_version=config_version,
    )


def build_intent(
    *,
    run_id: str,
    asof_date: date,
    sleeve: str,
    symbol: str,
    asset_class: AssetClass,
    target_weight: float,
    execution_phase: ExecutionPhase,
    multiplier: float,
    trace: TraceRef,
    metadata: dict[str, Any] | None = None,
) -> TargetIntent:
    """Build a single canonical TargetIntent."""
    return TargetIntent(
        intent_id=str(uuid.uuid4()),
        run_id=run_id,
        asof_date=asof_date,
        sleeve=sleeve,
        symbol=symbol,
        asset_class=asset_class,
        target_weight=target_weight,
        execution_phase=execution_phase,
        multiplier=multiplier,
        trace=trace,
        metadata=metadata or {},
    )


def build_decision(
    *,
    sleeve: str,
    run_id: str,
    asof_date: date,
    status: SleeveStatus,
    intents: tuple[TargetIntent, ...] = (),
    reason: str | None = None,
    diagnostics: dict[str, Any] | None = None,
    warnings: tuple[str, ...] = (),
    artifact_refs: dict[str, str] | None = None,
    control_snapshot_id: str = "",
    market_snapshot_id: str = "",
    portfolio_snapshot_id: str = "",
    started_at: datetime | None = None,
    generated_by: str = "",
) -> SleeveDecision:
    """Build a canonical SleeveDecision with current timestamps."""
    now = datetime.now(timezone.utc)
    return SleeveDecision(
        schema_version="sleeve_decision.v1",
        sleeve=sleeve,
        run_id=run_id,
        asof_date=asof_date,
        status=status,
        reason=reason,
        intents=intents,
        diagnostics=diagnostics or {},
        warnings=warnings,
        artifact_refs=artifact_refs or {},
        control_snapshot_id=control_snapshot_id,
        market_snapshot_id=market_snapshot_id,
        portfolio_snapshot_id=portfolio_snapshot_id,
        started_at=started_at or now,
        completed_at=now,
        generated_by=generated_by,
    )


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(with_app_metadata(obj), indent=2, sort_keys=True), encoding="utf-8")


def write_sleeve_decision(decision: SleeveDecision, day_root: Path) -> dict[str, str]:
    """Write canonical decision artifact. Returns artifact path dict."""
    from dataclasses import asdict

    sleeve_dir = day_root / "sleeves" / decision.sleeve
    sleeve_dir.mkdir(parents=True, exist_ok=True)
    decision_path = sleeve_dir / "decision.json"

    # Serialize — convert enums, dates, tuples to JSON-safe types
    payload = _decision_to_dict(decision)
    _write_json(decision_path, payload)

    return {"decision": str(decision_path)}


def write_compat_artifacts(decision: SleeveDecision, day_root: Path) -> dict[str, str]:
    """Derive and write legacy targets/signals compat artifacts."""
    targets_dir = day_root / "targets"
    signals_dir = day_root / "signals"
    targets_dir.mkdir(parents=True, exist_ok=True)
    signals_dir.mkdir(parents=True, exist_ok=True)

    tgt_path = targets_dir / f"{decision.sleeve}_targets.json"
    sig_path = signals_dir / f"{decision.sleeve}_signals.json"

    _write_json(tgt_path, sleeve_decision_to_targets_compat(decision))
    _write_json(sig_path, sleeve_decision_to_signals_compat(decision))

    return {"targets_compat": str(tgt_path), "signals_compat": str(sig_path)}


def _decision_to_dict(decision: SleeveDecision) -> dict[str, Any]:
    """Convert a SleeveDecision to a JSON-serializable dict."""
    intents = []
    for ti in decision.intents:
        intents.append({
            "intent_id": ti.intent_id,
            "run_id": ti.run_id,
            "asof_date": ti.asof_date.isoformat(),
            "sleeve": ti.sleeve,
            "symbol": ti.symbol,
            "asset_class": ti.asset_class.value,
            "target_weight": ti.target_weight,
            "execution_phase": ti.execution_phase.value,
            "multiplier": ti.multiplier,
            "metadata": ti.metadata,
            "trace": {
                "run_id": ti.trace.run_id,
                "sleeve": ti.trace.sleeve,
                "sleeve_run_id": ti.trace.sleeve_run_id,
                "source_artifact": ti.trace.source_artifact,
                "source_row_index": ti.trace.source_row_index,
                "control_snapshot_id": ti.trace.control_snapshot_id,
                "market_snapshot_id": ti.trace.market_snapshot_id,
                "portfolio_snapshot_id": ti.trace.portfolio_snapshot_id,
                "model_version": ti.trace.model_version,
                "config_version": ti.trace.config_version,
            },
        })

    return {
        "schema_version": decision.schema_version,
        "sleeve": decision.sleeve,
        "run_id": decision.run_id,
        "asof_date": decision.asof_date.isoformat(),
        "status": decision.status.value,
        "reason": decision.reason,
        "intents": intents,
        "diagnostics": decision.diagnostics,
        "warnings": list(decision.warnings),
        "artifact_refs": decision.artifact_refs,
        "control_snapshot_id": decision.control_snapshot_id,
        "market_snapshot_id": decision.market_snapshot_id,
        "portfolio_snapshot_id": decision.portfolio_snapshot_id,
        "started_at": decision.started_at.isoformat(),
        "completed_at": decision.completed_at.isoformat(),
        "generated_by": decision.generated_by,
    }


def legacy_rows_to_intents(
    rows: list[dict[str, Any]],
    *,
    run_id: str,
    asof_date: date,
    sleeve: str,
    trace: TraceRef,
    default_asset_class: AssetClass = AssetClass.EQUITY,
    default_execution_phase: ExecutionPhase = ExecutionPhase.INTRADAY,
    default_multiplier: float = 1.0,
) -> tuple[TargetIntent, ...]:
    """Convert legacy target rows into canonical TargetIntent tuple.

    This is the critical Phase 3.5 bridge: it wraps legacy VRP `orders`
    and legacy selector `targets` into canonical intents so that enabling
    ``FF_CANONICAL_SLEEVE_OUTPUTS`` does not change the active sleeve set.

    Rows that have a zero or missing target_weight are skipped.
    """
    intents: list[TargetIntent] = []
    for i, row in enumerate(rows):
        symbol = str(row.get("symbol", "")).strip()
        if not symbol:
            continue
        weight = float(row.get("target_weight", 0.0))
        if weight == 0.0:
            continue

        # Parse asset class
        raw_ac = str(row.get("asset_class", default_asset_class.value)).upper()
        try:
            asset_class = AssetClass(raw_ac)
        except ValueError:
            asset_class = default_asset_class

        # Parse execution phase
        raw_phase = row.get("execution_phase", default_execution_phase.value)
        if hasattr(raw_phase, "value"):
            raw_phase = raw_phase.value
        try:
            execution_phase = ExecutionPhase(str(raw_phase))
        except ValueError:
            execution_phase = default_execution_phase

        multiplier = float(row.get("multiplier", default_multiplier))
        if multiplier <= 0:
            multiplier = default_multiplier

        # Clamp weight to [-2.0, 2.0]
        weight = max(-2.0, min(2.0, weight))

        metadata = {}
        if "dte" in row:
            metadata["dte"] = int(row["dte"])
        if "score" in row:
            metadata["score"] = float(row["score"])
        if "side" in row:
            metadata["side"] = str(row["side"])

        intents.append(TargetIntent(
            intent_id=str(uuid.uuid4()),
            run_id=run_id,
            asof_date=asof_date,
            sleeve=sleeve,
            symbol=symbol,
            asset_class=asset_class,
            target_weight=weight,
            execution_phase=execution_phase,
            multiplier=multiplier,
            trace=TraceRef(
                run_id=trace.run_id,
                sleeve=trace.sleeve,
                sleeve_run_id=trace.sleeve_run_id,
                source_artifact=trace.source_artifact,
                source_row_index=i,
                control_snapshot_id=trace.control_snapshot_id,
                market_snapshot_id=trace.market_snapshot_id,
                portfolio_snapshot_id=trace.portfolio_snapshot_id,
                model_version=trace.model_version,
                config_version=trace.config_version,
            ),
            metadata=metadata,
        ))
    return tuple(intents)


def get_snapshot_ids(ctx: dict[str, Any]) -> tuple[str, str, str]:
    """Extract typed snapshot IDs from the job context dict."""
    ctrl = ctx.get("typed_control_snapshot")
    mkt = ctx.get("typed_market_snapshot")
    pf = ctx.get("typed_portfolio_snapshot")
    return (
        ctrl.snapshot_id if ctrl else ctx.get("control_snapshot_id", ""),
        mkt.snapshot_id if mkt else "",
        pf.snapshot_id if pf else "",
    )
