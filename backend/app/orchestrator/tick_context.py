from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TickContext:
    model_versions: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_model_version(
        self,
        model_key: str,
        *,
        model_name: str,
        model_version: str,
        endpoint_name: str,
        model_alias: str | None = None,
        latency_ms: float | None = None,
    ) -> None:
        if not model_version:
            return
        payload: dict[str, Any] = {
            "model_name": model_name,
            "model_version": str(model_version),
            "endpoint_name": endpoint_name,
        }
        if model_alias is not None:
            payload["model_alias"] = str(model_alias)
        if latency_ms is not None:
            payload["latency_ms"] = float(latency_ms)
        self.model_versions[model_key] = payload

