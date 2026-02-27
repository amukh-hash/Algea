from __future__ import annotations


def forward_realized_vol(close: list[float], idx: int, tenor: int) -> float | None:
    end = idx + tenor
    if end >= len(close):
        return None
    rs = []
    for i in range(idx + 1, end + 1):
        if close[i - 1] == 0:
            return None
        rs.append((close[i] / close[i - 1]) - 1.0)
    if not rs:
        return None
    return (sum(r * r for r in rs) / len(rs)) ** 0.5
