from __future__ import annotations

from .constraints import apply_caps
from .optimizer import normalize_long_short


class SelectorPortfolioConstructor:
    def __init__(self, per_symbol_cap: float = 0.05):
        self.per_symbol_cap = per_symbol_cap

    def construct(self, scores: dict[str, float]) -> list[dict]:
        weights = normalize_long_short(scores, top_n=min(5, max(1, len(scores) // 2)), gross_target=1.0)
        capped = apply_caps(weights, self.per_symbol_cap)
        return [{"symbol": s, "target_weight": round(w, 6)} for s, w in sorted(capped.items()) if abs(w) > 0]
