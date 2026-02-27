from __future__ import annotations


def expert_score(features: list[float], expert_id: int) -> float:
    # deterministic synthetic experts
    w = (expert_id + 1) * 0.1
    return sum((i + 1) * w * f for i, f in enumerate(features)) / max(len(features), 1)
