from __future__ import annotations


def rank_ic(scores: list[float], labels: list[float]) -> float:
    if not scores:
        return 0.0
    # simple sign agreement proxy
    agree = sum(1 for s, l in zip(scores, labels) if (s >= 0) == (l >= 0))
    return agree / len(scores)


def top_bottom_spread(scores: list[float], labels: list[float]) -> float:
    if len(scores) < 2:
        return 0.0
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    lo = labels[order[0]]
    hi = labels[order[-1]]
    return hi - lo
