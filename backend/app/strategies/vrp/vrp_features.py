from __future__ import annotations


def build_vrp_history(features_by_tenor: dict[int, dict], lookback: int = 20) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for t, feat in features_by_tenor.items():
        out[int(t)] = [dict(feat) for _ in range(lookback)]
    return out
