from __future__ import annotations


def mid_price(row: dict) -> float:
    b = float(row.get("bid", 0.0))
    a = float(row.get("ask", 0.0))
    if b > 0 and a > 0:
        return (a + b) / 2.0
    return float(row.get("mid", 0.0))
