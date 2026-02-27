from __future__ import annotations

from pydantic import BaseModel, Field


class SMoERankRequest(BaseModel):
    asof: str
    symbols: list[str]
    feature_matrix: list[list[float]]
    market_context: dict[str, float] = Field(default_factory=dict)
    model_alias: str = "prod"
    trace_id: str


class SMoERankResponse(BaseModel):
    model_name: str = "selector_smoe"
    model_version: str
    scores: dict[str, float]
    router_entropy_mean: float
    expert_utilization: dict[str, int]
    load_balance_score: float
    latency_ms: float
    warnings: list[str] = Field(default_factory=list)
