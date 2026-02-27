from __future__ import annotations

from pydantic import BaseModel, Field


class RLPolicyRequest(BaseModel):
    asof: str
    sleeve: str
    state: dict[str, float]
    proposal: dict[str, float | int | str]
    constraints: dict[str, float | int | bool]
    model_alias: str = "prod"
    trace_id: str


class RLPolicyResponse(BaseModel):
    model_name: str = "rl_policy"
    model_version: str
    size_multiplier: float
    veto: bool
    projected_multiplier: float
    projection_reason: str
    projection_applied: bool
    drift_score: float = 0.0
    ood_score: float = 0.0
    latency_ms: float
    warnings: list[str] = Field(default_factory=list)
