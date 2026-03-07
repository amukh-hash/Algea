from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class SleeveOutput(BaseModel):
    """Post-handler sleeve boundary contract.

    Compatibility mode accepts legacy artifact-only outputs during migration.
    """

    model_config = ConfigDict(extra="allow")

    status: str = "ok"
    artifacts: dict[str, Any] | list[Any] | None = None
    targets: list[dict[str, Any]] | None = None
    intents: list[dict[str, Any]] | None = None


class RiskDecisionReport(BaseModel):
    """Canonical risk decision artifact contract (risk_checks.json)."""

    model_config = ConfigDict(extra="allow")

    schema_version: str = "risk_decision.v1"
    status: str
    decision_id: str
    checked_at: str | None = None
    asof_date: str | None = None
    session: str | None = None
    policy_version: str = "risk_decision_policy.v1"
    input_contract_family: str = "targets_legacy"
    source_sleeves: list[str] = Field(default_factory=list)
    input_artifact_refs: dict[str, Any] = Field(default_factory=dict)
    generated_by: str = "handle_risk_checks_global"
    reason: str | None = None
    missing_sleeves: list[str] = Field(default_factory=list)
    inputs: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    limits: dict[str, Any] = Field(default_factory=dict)
    violations: list[dict[str, Any]] = Field(default_factory=list)


class PositionDeltaOrder(BaseModel):
    model_config = ConfigDict(extra="allow")

    symbol: str
    qty: int = Field(..., gt=0)
    side: Literal["BUY", "SELL"]
    type: str | None = None
    tif: str | None = None
    est_price: float | None = None
    est_notional: float | None = None


class PositionDeltaPlan(BaseModel):
    """Order plan contract produced by planner before broker placement."""

    model_config = ConfigDict(extra="allow")

    asof_date: str
    session: str
    mode: str
    dry_run: bool
    orders: list[PositionDeltaOrder] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)


class BrokerPosition(BaseModel):
    """Canonical broker position record (persisted field: quantity)."""

    model_config = ConfigDict(extra="allow")

    symbol: str
    quantity: float
    avg_cost: float = 0.0


class ExecutionLeg(BaseModel):
    model_config = ConfigDict(extra="allow")

    ticker: str
    qty: float
    side: str
    status: str
    broker_order_id: str | None = None


class ExecutionResult(BaseModel):
    """Contract for broker execution response returned from adapter."""

    model_config = ConfigDict(extra="allow")

    status: str
    order_count: int
    routed: list[ExecutionLeg] = Field(default_factory=list)
    account: str | None = None
