from __future__ import annotations

from pydantic import BaseModel, Field


class VolSurfaceRequest(BaseModel):
    asof: str
    underlying_symbol: str
    tenors: list[int]
    history: dict[int, list[dict]]
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    model_alias: str = "prod"
    trace_id: str


class VolSurfaceResponse(BaseModel):
    model_name: str = "vol_surface"
    model_version: str
    predicted_rv: dict[int, dict[str, float]]
    uncertainty: dict[int, float]
    ood_score: float
    drift_score: float
    latency_ms: float
    warnings: list[str] = Field(default_factory=list)
