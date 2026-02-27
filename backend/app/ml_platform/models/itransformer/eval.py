from __future__ import annotations


def rank_ic(scores: list[float], labels: list[float]) -> float:
    if not scores:
        return 0.0
    agree = sum(1 for s, l in zip(scores, labels) if (s >= 0) == (l >= 0))
    return agree / len(scores)
