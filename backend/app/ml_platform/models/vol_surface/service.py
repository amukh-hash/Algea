from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ...drift.detectors import tenor_drift_score
from ...replay.hashes import hash_payload
from ...replay.trace import write_trace
from .model import VolSurfaceForecaster
from .types import VolSurfaceRequest, VolSurfaceResponse


class VolSurfaceService:
    def __init__(self, loader, trace_root: Path, timeout_ms: int = 500, device: str = "cpu"):
        self.device = device
        self.loader = loader
        self.trace_root = trace_root
        self.timeout_ms = timeout_ms
        self._loaded: dict[str, dict] = {}
        self._cache: dict[str, VolSurfaceResponse] = {}

    def load_alias(self, alias: str = "prod") -> None:
        self._loaded[alias] = self.loader.load_alias(alias)

    def _bundle(self, alias: str) -> dict:
        if alias not in self._loaded:
            self.load_alias(alias)
        return self._loaded[alias]

    def forecast(self, req: VolSurfaceRequest) -> VolSurfaceResponse:
        started = time.perf_counter()
        bundle = self._bundle(req.model_alias)
        key = hashlib.sha256(json.dumps(req.model_dump(), sort_keys=True).encode()).hexdigest()
        if key in self._cache:
            return self._cache[key]

        model = VolSurfaceForecaster(hidden_size=int(bundle["config"].get("hidden_size", 16)))
        preds = model.forecast(req.history, req.quantiles)
        uncertainty = {}
        for t, qs in preds.items():
            q10 = qs.get("0.10", 0.0)
            q90 = qs.get("0.90", 0.0)
            uncertainty[int(t)] = abs(q90 - q10)
        drift = tenor_drift_score(req.history, bundle.get("drift_baseline", {}))
        elapsed = (time.perf_counter() - started) * 1000
        if elapsed > self.timeout_ms:
            raise TimeoutError(f"vol_surface timeout {elapsed:.1f}>{self.timeout_ms}")
        resp = VolSurfaceResponse(
            model_version=bundle["model_version"],
            predicted_rv={int(k): v for k, v in preds.items()},
            uncertainty=uncertainty,
            ood_score=drift,
            drift_score=drift,
            latency_ms=elapsed,
        )
        write_trace(
            self.trace_root,
            req.trace_id,
            {
                "trace_id": req.trace_id,
                "model_name": "vol_surface",
                "model_version": bundle["model_version"],
                "request_hash": hash_payload(req.model_dump()),
                "output_hash": hash_payload(resp.model_dump()),
                "uncertainty": resp.uncertainty,
                "drift_score": drift,
                "ood_score": drift,
                "latency_ms": elapsed,
            },
        )
        self._cache[key] = resp
        return resp
