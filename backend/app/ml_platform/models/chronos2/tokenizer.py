from __future__ import annotations


def normalize_series(series: list[float]) -> list[float]:
    if not series:
        return []
    mean = sum(series) / len(series)
    centered = [x - mean for x in series]
    scale = max((abs(x) for x in centered), default=1.0) or 1.0
    return [x / scale for x in centered]
