from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class TSFMRequest(BaseModel):
    asof: str
    series: list[float]
    timestamps: list[str] | None = None
    freq: str
    prediction_length: int
    context_length: int | None = None
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9])
    instrument_id: str
    trace_id: str
    model_alias: str = "prod"


class TSFMResponse(BaseModel):
    model_name: str = "chronos2"
    model_version: str
    forecast: dict[str, list[float]]
    uncertainty: dict[str, float]
    ood_score: float | None = None
    calibration_score: float | None = None
    latency_ms: float
    warnings: list[str] = Field(default_factory=list)


class Chronos2ModelBundle(BaseModel):
    model_name: str = "chronos2"
    model_version: str
    config: dict[str, Any]
    calibration: dict[str, Any]
    drift_baseline: dict[str, Any]
    artifact_dir: str
