from __future__ import annotations


def normalize_long_short(scores: dict[str, float], top_n: int = 5, gross_target: float = 1.0) -> dict[str, float]:
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    longs = ranked[:top_n]
    shorts = ranked[-top_n:] if len(ranked) >= top_n else []
    out: dict[str, float] = {}
    if longs:
        w = gross_target * 0.5 / len(longs)
        for s, _ in longs:
            out[s] = w
    if shorts:
        w = gross_target * 0.5 / len(shorts)
        for s, _ in shorts:
            out[s] = -w
    return out
