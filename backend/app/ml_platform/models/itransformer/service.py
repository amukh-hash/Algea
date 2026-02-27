from __future__ import annotations

import json
import time
from pathlib import Path

from ...drift.detectors import simple_drift_score
from ...replay.hashes import hash_payload
from ...replay.trace import write_trace
from .model import ITransformerModel
from .types import ITransformerSignalRequest, ITransformerSignalResponse


class ITransformerService:
    def __init__(self, loader, trace_root: Path, timeout_ms: int = 500):
        self.loader = loader
        self.trace_root = trace_root
        self.timeout_ms = timeout_ms
        self._loaded: dict[str, dict] = {}

    def load_alias(self, alias: str = "prod") -> None:
        self._loaded[alias] = self.loader.load_alias(alias)

    def _bundle(self, alias: str) -> dict:
        if alias not in self._loaded:
            self.load_alias(alias)
        return self._loaded[alias]

    def signal(self, req: ITransformerSignalRequest) -> ITransformerSignalResponse:
        started = time.perf_counter()
        bundle = self._bundle(req.model_alias)
        model = ITransformerModel(hidden_size=int(bundle["config"].get("hidden_size", 32)))
        raw_scores, corr_regime = model.signal(req.feature_matrix)
        elapsed = (time.perf_counter() - started) * 1000
        if elapsed > self.timeout_ms:
            raise TimeoutError("itransformer timeout")
        scores = {s: raw_scores[i] for i, s in enumerate(req.symbols)}
        live_mean = sum(raw_scores) / max(len(raw_scores), 1)
        drift = simple_drift_score(float(bundle["drift_baseline"].get("score_mean", 0.0)), live_mean)
        resp = ITransformerSignalResponse(
            model_version=bundle["model_version"],
            scores=scores,
            uncertainty=1.0 / (1.0 + abs(live_mean)),
            correlation_regime=corr_regime,
            latency_ms=elapsed,
        )
        write_trace(
            self.trace_root,
            req.trace_id,
            {
                "trace_id": req.trace_id,
                "model_name": "itransformer",
                "model_version": bundle["model_version"],
                "request_hash": hash_payload(req.model_dump()),
                "output_hash": hash_payload(resp.model_dump()),
                "drift_score": drift,
                "latency_ms": elapsed,
            },
        )
        return resp
