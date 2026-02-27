from __future__ import annotations


def panel_label_fwd_ret(close: list[float], idx: int, horizon: int = 5) -> float | None:
    j = idx + horizon
    if idx < 0 or j >= len(close) or close[idx] == 0:
        return None
    return (close[j] / close[idx]) - 1.0
