"""Canonical contracts for Intent Supremacy architecture.

Canonical dataclass contracts are the **authoritative internal runtime
contracts** for all new orchestration work.  Legacy Pydantic models
(``backend.app.core.schemas``, ``backend.app.contracts.models``) and
legacy ``targets/*.json`` / ``signals/*.json`` artifact shapes are
**compatibility interfaces only** and will be retired in later migration
phases.

All production-facing sleeve output, risk evaluation, and planning types.
Frozen dataclasses enforce immutability throughout the pipeline.

State Machine — ``SleeveStatus`` invariant table
-------------------------------------------------

::

    OK        -> intents MUST be non-empty (valid trade produced)
    HALTED    -> intents MUST be empty   (valid data, intentional no-trade)
    FAILED    -> intents MUST be empty   (sleeve error, fail-closed)
    DISABLED  -> intents MUST be empty   (feature off, not evaluated)

Collation inclusion policy
--------------------------

::

    OK        -> include intents          (counted as active sleeve)
    HALTED    -> 0 intents, evaluated     (does NOT block collation)
    DISABLED  -> 0 intents, excluded      (does NOT block collation)
    FAILED    -> hard fail collation      (blocks entire pipeline)
    MISSING   -> hard fail if expected    (blocks entire pipeline)

Mandatory observability fields per artifact
-------------------------------------------

::

    SleeveDecision:
      run_id, sleeve, status, generated_by, market_snapshot_id,
      portfolio_snapshot_id, diagnostics.source_branch

    canonical_intents.json:
      collation_id, run_id, asof_date, schema_version,
      sleeve_statuses (per-sleeve inclusion field),
      inclusion_policy, total_intents

    risk_checks.json:
      decision_id, input_contract_family, collation_id (if canonical),
      source_sleeves, input_artifact_refs, flags

    order_plan.json:
      planning_input_family, collation_id (if canonical),
      symbol_trace_refs
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Literal


# ── Enums ──────────────────────────────────────────────────────────────


class SleeveStatus(str, Enum):
    """Canonical sleeve execution outcome.

    OK       — valid data, produced tradeable intents.
    HALTED   — valid data, valid execution, intentional no-trade.
    FAILED   — data/system error, fail-closed, no intents.
    DISABLED — feature flag off, not evaluated, no intents.
    """
    OK = "ok"
    HALTED = "halted"
    DISABLED = "disabled"
    FAILED = "failed"


class AssetClass(str, Enum):
    EQUITY = "EQUITY"
    FUTURE = "FUTURE"
    OPTION = "OPTION"


class ExecutionPhase(str, Enum):
    """Market micro-structure phases that determine when orders are routed."""
    PREMARKET = "premarket"
    AUCTION_OPEN = "auction_open"
    FUTURES_OPEN = "futures_open"
    INTRADAY = "intraday"
    COMMODITY_CLOSE = "commodity_close"
    AUCTION_CLOSE = "auction_close"
    FUTURES_CLOSE = "futures_close"


# ── Tracing ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TraceRef:
    """Full lineage trace for a single intent.

    Every intent carries its complete provenance so that any downstream
    consumer (risk, planner, audit) can trace back to the exact run,
    snapshot versions, and source artifact row.
    """
    run_id: str
    sleeve: str
    sleeve_run_id: str
    source_artifact: str
    source_row_index: int | None
    control_snapshot_id: str
    market_snapshot_id: str
    portfolio_snapshot_id: str
    policy_version: str | None = None
    model_version: str | None = None
    config_version: str | None = None


# ── Intent ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TargetIntent:
    """Canonical sleeve position intent with full tracing.

    This is the **authoritative internal runtime contract** for sleeve
    outputs.  The legacy Pydantic ``TargetIntent`` in
    ``backend.app.core.schemas`` is an input-compatibility parser only.
    """
    intent_id: str
    run_id: str
    asof_date: date
    sleeve: str
    symbol: str
    asset_class: AssetClass
    target_weight: float
    execution_phase: ExecutionPhase
    multiplier: float
    trace: TraceRef
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (-2.0 <= self.target_weight <= 2.0):
            raise ValueError(
                f"target_weight {self.target_weight} outside [-2.0, 2.0]"
            )
        if self.multiplier <= 0:
            raise ValueError(f"multiplier must be positive, got {self.multiplier}")
        if not self.symbol or not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if not self.intent_id:
            raise ValueError("intent_id must be non-empty")


# ── Sleeve Decision ───────────────────────────────────────────────────


@dataclass(frozen=True)
class SleeveDecision:
    """Canonical sleeve output contract.

    Every enabled sleeve handler must return exactly one of these.

    State invariant table::

        OK        -> intents MUST be non-empty
        HALTED    -> intents MUST be empty
        FAILED    -> intents MUST be empty
        DISABLED  -> intents MUST be empty

    ``HALTED`` means: valid data was available, the sleeve executed
    its logic successfully, and the result is an intentional no-trade.
    Diagnostics and warnings MAY be populated.
    """
    schema_version: Literal["sleeve_decision.v1"]
    sleeve: str
    run_id: str
    asof_date: date
    status: SleeveStatus
    reason: str | None
    intents: tuple[TargetIntent, ...]
    diagnostics: dict[str, Any]
    warnings: tuple[str, ...]
    artifact_refs: dict[str, str]
    control_snapshot_id: str
    market_snapshot_id: str
    portfolio_snapshot_id: str
    started_at: datetime
    completed_at: datetime
    generated_by: str

    def __post_init__(self) -> None:
        if self.status == SleeveStatus.OK and not self.intents:
            raise ValueError(
                f"sleeve '{self.sleeve}' status=OK but intents is empty; "
                "use HALTED for valid-no-trade"
            )
        if self.status == SleeveStatus.HALTED and self.intents:
            raise ValueError(
                f"sleeve '{self.sleeve}' status=HALTED but intents is non-empty; "
                "HALTED means valid-data-but-intentional-no-trade"
            )
        if self.status == SleeveStatus.FAILED and self.intents:
            raise ValueError(
                f"sleeve '{self.sleeve}' status=FAILED but intents is non-empty"
            )
        if self.status == SleeveStatus.DISABLED and self.intents:
            raise ValueError(
                f"sleeve '{self.sleeve}' status=DISABLED but intents is non-empty"
            )


# ── Risk Decision ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class Violation:
    """A single risk check violation."""
    code: str
    message: str
    severity: Literal["error", "warning"]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskDecisionReport:
    """Canonical risk evaluation output (v2)."""
    schema_version: Literal["risk_decision.v2"]
    decision_id: str
    run_id: str
    asof_date: date
    status: Literal["ok", "failed", "halted"]
    control_snapshot_id: str
    market_snapshot_id: str
    portfolio_snapshot_id: str
    input_family: Literal["canonical_intents"]
    source_sleeves: tuple[str, ...]
    violations: tuple[Violation, ...]
    exposures: dict[str, Any]
    limits: dict[str, Any]
    diagnostics: dict[str, Any]
    generated_by: str


# ── Planning ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PlannedOrder:
    """A single planned order derived from canonical intents."""
    order_id: str
    symbol: str
    side: Literal["BUY", "SELL"]
    qty: int
    est_price: float
    est_notional: float
    intent_refs: tuple[str, ...]
    price_source: str

    def __post_init__(self) -> None:
        if self.qty <= 0:
            raise ValueError(f"qty must be positive, got {self.qty}")
        if self.est_price < 0:
            raise ValueError(f"est_price must be non-negative, got {self.est_price}")


@dataclass(frozen=True)
class PositionDeltaPlan:
    """Canonical order plan (v2)."""
    schema_version: Literal["position_delta_plan.v2"]
    plan_id: str
    run_id: str
    asof_date: date
    control_snapshot_id: str
    market_snapshot_id: str
    portfolio_snapshot_id: str
    source_sleeves: tuple[str, ...]
    orders: tuple[PlannedOrder, ...]
    diagnostics: dict[str, Any]
    generated_by: str
