from __future__ import annotations

from dataclasses import dataclass

from .experts import expert_score
from .router import topk_router


@dataclass
class SMoEConfig:
    n_experts: int = 4
    top_k: int = 1


class SMoERankerModel:
    def __init__(self, cfg: SMoEConfig | None = None):
        self.cfg = cfg or SMoEConfig()

    def forward_row(self, features: list[float], market_context: list[float]) -> dict:
        ctx = sum((j + 1) * v for j, v in enumerate(market_context))
        logits = [sum(features[:4]) * (i + 1) * 0.05 + ctx * (i + 1) * 0.03 for i in range(self.cfg.n_experts)]
        chosen, probs, entropy = topk_router(logits, k=self.cfg.top_k)
        score = sum(expert_score(features, i) * probs[i] for i in chosen)
        return {
            "score": float(score),
            "router_entropy": float(entropy),
            "expert_id": int(chosen[0]),
            "expert_probs": probs,
            "uncertainty": float(1.0 / (1.0 + abs(score))),
        }
