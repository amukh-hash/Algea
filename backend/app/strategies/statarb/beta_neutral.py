from __future__ import annotations


def beta_neutralize(weights: dict[str, float]) -> dict[str, float]:
    net = sum(weights.values())
    adj = net / max(len(weights), 1)
    return {s: w - adj for s, w in weights.items()}
