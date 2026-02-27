from __future__ import annotations

import json
import time
from pathlib import Path

from ...replay.hashes import hash_payload
from ...replay.trace import write_trace
from ..selector_master.context_encoder import encode_market_context, validate_market_context
from .model import SMoEConfig, SMoERankerModel
from .types import SMoERankRequest, SMoERankResponse


class SMoEService:
    def __init__(self, loader, trace_root: Path, microbatch_size: int = 64):
        self.loader = loader
        self.trace_root = trace_root
        self.microbatch_size = microbatch_size
        self._loaded: dict[str, dict] = {}

    def load_alias(self, alias: str = "prod") -> None:
        self._loaded[alias] = self.loader.load_alias(alias)

    def _bundle(self, alias: str) -> dict:
        if alias not in self._loaded:
            self.load_alias(alias)
        return self._loaded[alias]

    def rank(self, req: SMoERankRequest) -> SMoERankResponse:
        started = time.perf_counter()
        bundle = self._bundle(req.model_alias)
        cfg = SMoEConfig(**bundle["config"])
        model = SMoERankerModel(cfg)
        if bool(bundle["config"].get("require_context", False)):
            validate_market_context(req.market_context)
        ctx = encode_market_context(req.market_context)

        scores: dict[str, float] = {}
        expert_util: dict[int, int] = {i: 0 for i in range(cfg.n_experts)}
        entropies: list[float] = []
        noctx_entropies: list[float] = []
        noctx_scores: list[float] = []
        ctx_scores: list[float] = []

        for i in range(0, len(req.symbols), self.microbatch_size):
            chunk_syms = req.symbols[i : i + self.microbatch_size]
            chunk_feats = req.feature_matrix[i : i + self.microbatch_size]
            for s, f in zip(chunk_syms, chunk_feats):
                out = model.forward_row(f, ctx)
                out_noctx = model.forward_row(f, [0.0 for _ in ctx])
                scores[s] = out["score"]
                entropies.append(out["router_entropy"])
                noctx_entropies.append(out_noctx["router_entropy"])
                noctx_scores.append(out_noctx["score"])
                ctx_scores.append(out["score"])
                expert_util[out["expert_id"]] = expert_util.get(out["expert_id"], 0) + 1

        total = sum(expert_util.values()) or 1
        target = 1.0 / max(cfg.n_experts, 1)
        load_balance = sum(abs((expert_util.get(i, 0) / total) - target) for i in range(cfg.n_experts))
        collapse_score = max(expert_util.values()) / total
        context_sensitivity = sum(abs(a - b) for a, b in zip(ctx_scores, noctx_scores)) / max(len(ctx_scores), 1)
        sector_bucket = "risk_on" if req.market_context.get("sector_rel_strength", 0.0) >= 0 else "risk_off"
        specialization = {sector_bucket: {str(k): int(v) for k, v in expert_util.items()}}
        elapsed = (time.perf_counter() - started) * 1000
        resp = SMoERankResponse(
            model_version=bundle["model_version"],
            scores=scores,
            router_entropy_mean=sum(entropies) / max(len(entropies), 1),
            expert_utilization={str(k): v for k, v in expert_util.items()},
            load_balance_score=load_balance,
            context_sensitivity_score=context_sensitivity,
            expert_collapse_score=collapse_score,
            specialization_by_bucket=specialization,
            latency_ms=elapsed,
        )
        write_trace(
            self.trace_root,
            req.trace_id,
            {
                "trace_id": req.trace_id,
                "model_name": "selector_smoe",
                "model_version": bundle["model_version"],
                "request_hash": hash_payload(req.model_dump()),
                "output_hash": hash_payload(resp.model_dump()),
                "router_entropy_mean": resp.router_entropy_mean,
                "expert_utilization": resp.expert_utilization,
                "context_sensitivity_score": resp.context_sensitivity_score,
                "expert_collapse_score": resp.expert_collapse_score,
                "latency_ms": elapsed,
            },
        )
        return resp
