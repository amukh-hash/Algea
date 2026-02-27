from __future__ import annotations

import math


def softmax(xs: list[float]) -> list[float]:
    if not xs:
        return []
    m = max(xs)
    ex = [math.exp(x - m) for x in xs]
    s = sum(ex) or 1.0
    return [e / s for e in ex]


def topk_router(logits: list[float], k: int = 1) -> tuple[list[int], list[float], float]:
    probs = softmax(logits)
    ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    chosen = ranked[:k]
    entropy = -sum(p * math.log(max(p, 1e-12)) for p in probs)
    return chosen, probs, entropy
