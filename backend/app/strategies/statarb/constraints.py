from __future__ import annotations


def cap_weights(weights: dict[str, float], cap: float = 0.05) -> dict[str, float]:
    return {s: max(-cap, min(cap, w)) for s, w in weights.items()}
