from __future__ import annotations

from .beta_neutral import beta_neutralize
from .constraints import cap_weights


def build_statarb_targets(scores: dict[str, float], top_k: int = 3) -> list[dict]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1])
    k = min(top_k, max(1, len(ranked) // 2))
    longs = ranked[:k]
    shorts = ranked[-k:]
    weights = {}
    if longs:
        wl = 0.5 / len(longs)
        for s, _ in longs:
            weights[s] = wl
    if shorts:
        ws = -0.5 / len(shorts)
        for s, _ in shorts:
            weights[s] = ws
    weights = cap_weights(beta_neutralize(weights), cap=0.05)
    return [{"symbol": s, "target_weight": round(w, 6)} for s, w in sorted(weights.items()) if abs(w) > 0]
