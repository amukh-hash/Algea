from __future__ import annotations


def build_series_features(prices: list[float], context_length: int) -> list[float]:
    return prices[-context_length:]
