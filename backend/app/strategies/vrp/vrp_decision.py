from __future__ import annotations


def choose_tenor(edge_by_tenor: dict[int, float], uncertainty: dict[int, float], edge_threshold: float, uncertainty_threshold: float) -> int | None:
    candidates = [t for t, e in edge_by_tenor.items() if e > edge_threshold and uncertainty.get(t, 999.0) < uncertainty_threshold]
    if not candidates:
        return None
    return sorted(candidates, key=lambda t: edge_by_tenor[t], reverse=True)[0]
