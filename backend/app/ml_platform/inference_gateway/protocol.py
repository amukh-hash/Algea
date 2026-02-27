from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class InferenceRequestBase:
    asof: datetime
    universe_id: str
    features_hash: str
    model_alias: str = "prod"
    trace_id: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["asof"] = self.asof.isoformat()
        return data


@dataclass
class InferenceResponse:
    model_name: str
    model_version: str
    outputs: dict[str, Any]
    uncertainty: float
    calibration_score: float
    ood_score: float
    latency_ms: float
    warnings: list[str] = field(default_factory=list)
