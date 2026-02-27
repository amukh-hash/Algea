from __future__ import annotations


def realized_vol(close: list[float], window: int) -> float:
    if len(close) <= window:
        return 0.0
    rs = []
    for i in range(len(close) - window, len(close)):
        if i <= 0 or close[i - 1] == 0:
            continue
        rs.append((close[i] / close[i - 1]) - 1.0)
    if not rs:
        return 0.0
    return (sum(r * r for r in rs) / len(rs)) ** 0.5
