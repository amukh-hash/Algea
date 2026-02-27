from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from ...drift.detectors import zscore_ood_score
from ...replay.hashes import hash_payload
from ...replay.trace import write_trace
from .adapter import deterministic_quantile_forecast, summarize_uncertainty
from .loader import Chronos2Loader
from .types import TSFMRequest, TSFMResponse


class Chronos2Service:
    def __init__(self, loader: Chronos2Loader, trace_root: Path, timeout_ms: int = 500):
        self.loader = loader
        self.trace_root = trace_root
        self.timeout_ms = timeout_ms
        self._cache: dict[str, TSFMResponse] = {}
        self._model_by_alias: dict[str, object] = {}

    def load_alias(self, alias: str = "prod") -> None:
        self._model_by_alias[alias] = self.loader.load_alias(alias)

    def ready(self, alias: str = "prod") -> bool:
        return alias in self._model_by_alias

    def _bundle(self, alias: str):
        if alias not in self._model_by_alias:
            self.load_alias(alias)
        return self._model_by_alias[alias]

    def forecast(self, req: TSFMRequest) -> TSFMResponse:
        started = time.perf_counter()
        bundle = self._bundle(req.model_alias)
        key = hashlib.sha256(
            json.dumps(
                {
                    "mv": bundle.model_version,
                    "inst": req.instrument_id,
                    "series": req.series,
                    "pred": req.prediction_length,
                    "q": req.quantiles,
                },
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()
        if key in self._cache:
            return self._cache[key]

        forecast = deterministic_quantile_forecast(req.series, req.prediction_length, req.quantiles)
        uncertainty = summarize_uncertainty(forecast)
        ood = zscore_ood_score(req.series, bundle.drift_baseline)
        calibration = float(bundle.calibration.get("calibration_score", 0.5))
        elapsed = (time.perf_counter() - started) * 1000
        if elapsed > self.timeout_ms:
            raise TimeoutError(f"chronos2 timeout {elapsed:.2f}ms>{self.timeout_ms}ms")
        resp = TSFMResponse(
            model_version=bundle.model_version,
            forecast=forecast,
            uncertainty=uncertainty,
            ood_score=ood,
            calibration_score=calibration,
            latency_ms=elapsed,
        )
        trace_payload = {
            "trace_id": req.trace_id,
            "model_name": "chronos2",
            "model_version": bundle.model_version,
            "model_alias": req.model_alias,
            "request_hash": hash_payload(req.model_dump()),
            "output_hash": hash_payload(resp.model_dump()),
            "latency_ms": elapsed,
            "instrument_id": req.instrument_id,
        }
        write_trace(self.trace_root, req.trace_id, trace_payload)
        self._cache[key] = resp
        return resp
