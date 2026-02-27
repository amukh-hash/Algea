from __future__ import annotations

from math import sqrt


def clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def winsorize(values: list[float], lower_q: float = 0.05, upper_q: float = 0.95) -> list[float]:
    if not values:
        return []
    s = sorted(values)
    lo = s[int((len(s) - 1) * lower_q)]
    hi = s[int((len(s) - 1) * upper_q)]
    return [clip(v, lo, hi) for v in values]


def zscore(values: list[float]) -> list[float]:
    if not values:
        return []
    m = sum(values) / len(values)
    var = sum((v - m) ** 2 for v in values) / max(len(values), 1)
    sd = sqrt(var) if var > 0 else 1.0
    return [(v - m) / sd for v in values]
