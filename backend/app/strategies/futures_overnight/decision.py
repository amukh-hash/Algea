from __future__ import annotations


def should_trade(median_path: list[float], uncertainty_iqr_mean: float, min_edge: float = 0.001, max_iqr: float = 100.0) -> bool:
    if not median_path:
        return False
    edge = median_path[-1] - median_path[0]
    return abs(edge) >= min_edge and uncertainty_iqr_mean <= max_iqr
