from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from pathlib import Path

from ...drift.detectors import simple_drift_score
from ...replay.hashes import hash_payload
from ...replay.trace import write_trace
from ...rl.action_projection import project_action
from .model import RLPolicyModel
from .types import RLPolicyRequest, RLPolicyResponse


class RLPolicyService:
    def __init__(self, loader, trace_root: Path, timeout_ms: int = 120, cache_size: int = 1024, device: str = "cpu"):
        self.device = device
        self.loader = loader
        self.trace_root = trace_root
        self.timeout_ms = timeout_ms
        self.cache_size = cache_size
        self._loaded: dict[str, dict] = {}
        self._cache: OrderedDict[str, RLPolicyResponse] = OrderedDict()

    def load_alias(self, alias: str = "prod") -> None:
        self._loaded[alias] = self.loader.load_alias(alias)

    def _bundle(self, alias: str) -> dict:
        if alias not in self._loaded:
            self.load_alias(alias)
        return self._loaded[alias]

    def _cache_get(self, key: str) -> RLPolicyResponse | None:
        value = self._cache.get(key)
        if value is not None:
            self._cache.move_to_end(key)
        return value

    def _cache_put(self, key: str, value: RLPolicyResponse) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def policy_act(self, req: RLPolicyRequest) -> RLPolicyResponse:
        started = time.perf_counter()
        bundle = self._bundle(req.model_alias)
        state_hash = hashlib.sha256(json.dumps(req.state, sort_keys=True).encode()).hexdigest()
        proposal_hash = hashlib.sha256(json.dumps(req.proposal, sort_keys=True).encode()).hexdigest()
        constraints_hash = hashlib.sha256(json.dumps(req.constraints, sort_keys=True).encode()).hexdigest()
        key = f"{bundle['model_version']}|{req.sleeve}|{state_hash}|{proposal_hash}|{constraints_hash}"
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        model = RLPolicyModel(hidden_size=int(bundle["config"].get("hidden_size", 32)))
        vector = [float(req.state[k]) for k in sorted(req.state.keys())]
        size_multiplier, veto, confidence = model.act(vector)
        projected, reason = project_action(
            {"size_multiplier": size_multiplier, "veto": veto},
            req.constraints,
            req.proposal,
            req.state,
        )
        elapsed = (time.perf_counter() - started) * 1000
        if elapsed > self.timeout_ms:
            raise TimeoutError(f"rl_policy timeout {elapsed:.1f}>{self.timeout_ms}")
        drift_score = simple_drift_score(float(bundle["drift_baseline"].get("state_mean", 0.0)), sum(vector) / max(len(vector), 1))
        resp = RLPolicyResponse(
            model_version=bundle["model_version"],
            size_multiplier=size_multiplier,
            veto=veto,
            projected_multiplier=float(projected["size_multiplier"]),
            projection_reason=reason,
            projection_applied=reason != "",
            drift_score=drift_score,
            ood_score=drift_score,
            latency_ms=elapsed,
            warnings=[] if confidence > 0.05 else ["low_confidence"],
        )
        write_trace(
            self.trace_root,
            req.trace_id,
            {
                "trace_id": req.trace_id,
                "model_name": "rl_policy",
                "model_version": bundle["model_version"],
                "request_hash": hash_payload(req.model_dump()),
                "request_payload": req.model_dump(),
                "output_hash": hash_payload(resp.model_dump()),
                "state_hash": state_hash,
                "proposal_hash": proposal_hash,
                "constraints_hash": constraints_hash,
                "latency_ms": elapsed,
                "veto": resp.veto,
                "size_multiplier": resp.size_multiplier,
                "projected_multiplier": resp.projected_multiplier,
                "projection_applied": resp.projection_applied,
                "projection_reason": resp.projection_reason,
                "drift_score": resp.drift_score,
                "ood_score": resp.ood_score,
            },
        )
        self._cache_put(key, resp)
        return resp
