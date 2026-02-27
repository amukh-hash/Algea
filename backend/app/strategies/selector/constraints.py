from __future__ import annotations


def apply_caps(weights: dict[str, float], per_symbol_cap: float) -> dict[str, float]:
    return {s: max(-per_symbol_cap, min(per_symbol_cap, w)) for s, w in weights.items()}
