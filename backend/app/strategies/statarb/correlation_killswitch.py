from __future__ import annotations


def correlation_break(correlation_regime: float, threshold: float = 2.0) -> bool:
    return correlation_regime > threshold
