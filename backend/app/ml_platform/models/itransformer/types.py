from __future__ import annotations

from pydantic import BaseModel, Field


class ITransformerSignalRequest(BaseModel):
    asof: str
    symbols: list[str]
    feature_matrix: list[list[float]]
    model_alias: str = "prod"
    trace_id: str


class ITransformerSignalResponse(BaseModel):
    model_name: str = "itransformer"
    model_version: str
    scores: dict[str, float]
    uncertainty: float
    correlation_regime: float
    latency_ms: float
    warnings: list[str] = Field(default_factory=list)
