from __future__ import annotations


def fwd_return(prices: list[float], idx: int, horizon: int) -> float | None:
    j = idx + horizon
    if idx < 0 or j >= len(prices):
        return None
    p0 = float(prices[idx])
    p1 = float(prices[j])
    if p0 == 0:
        return None
    return (p1 / p0) - 1.0
