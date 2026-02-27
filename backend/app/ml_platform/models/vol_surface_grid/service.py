from __future__ import annotations

import time
from pathlib import Path

from ...replay.hashes import hash_payload
from ...replay.trace import write_trace
from .model import VolSurfaceGridForecaster
from .types import VolSurfaceGridRequest, VolSurfaceGridResponse


class VolSurfaceGridService:
    def __init__(self, loader, trace_root: Path, device: str = "cpu"):
        self.device = device
        self.loader = loader
        self.trace_root = trace_root
        self._loaded: dict[str, dict] = {}

    def load_alias(self, alias: str = "prod") -> None:
        self._loaded[alias] = self.loader.load_alias(alias)

    def _bundle(self, alias: str) -> dict:
        if alias not in self._loaded:
            self.load_alias(alias)
        return self._loaded[alias]

    def forecast(self, req: VolSurfaceGridRequest) -> VolSurfaceGridResponse:
        start = time.perf_counter()
        b = self._bundle(req.model_alias)
        model = VolSurfaceGridForecaster(scale=float(b["config"].get("scale", 0.05)))
        pred, unc, drift = model.forecast(req.grid_history)
        mask_cov = 0.0
        if req.grid_history:
            iv = req.grid_history[-1].get("iv", {})
            expected = max(len(iv), 1)
            mask_cov = len([k for k, v in iv.items() if v is not None]) / expected
        elapsed = (time.perf_counter() - start) * 1000
        resp = VolSurfaceGridResponse(model_version=b["model_version"], grid_forecast=pred, uncertainty_proxy=unc, drift_score=drift, mask_coverage=mask_cov, latency_ms=elapsed)
        write_trace(self.trace_root, req.trace_id, {"trace_id": req.trace_id, "model_name": "vol_surface_grid", "model_version": b["model_version"], "request_hash": hash_payload(req.model_dump()), "output_hash": hash_payload(resp.model_dump()), "mask_coverage": mask_cov, "uncertainty_proxy": unc, "drift_score": drift})
        return resp
