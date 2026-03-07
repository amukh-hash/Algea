from __future__ import annotations

from pydantic import BaseModel, Field


class VolSurfaceGridRequest(BaseModel):
    asof: str
    underlying_symbol: str
    grid_history: list[dict]
    model_alias: str = "prod"
    trace_id: str


class VolSurfaceGridResponse(BaseModel):
    model_name: str = "vol_surface_grid"
    model_version: str
    grid_forecast: float | dict[str, float]
    uncertainty_proxy: float
    drift_score: float
    mask_coverage: float
    latency_ms: float
    warnings: list[str] = Field(default_factory=list)
