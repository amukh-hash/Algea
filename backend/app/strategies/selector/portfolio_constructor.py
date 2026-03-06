from __future__ import annotations

from .constraints import apply_caps
from .optimizer import normalize_long_short


def apply_hysteresis_buffer(
    raw_scores: dict[str, float],
    current_positions: list[str],
    buffer_multiplier: float = 1.15,
) -> dict[str, float]:
    """Multiply scores of existing holdings to prevent trivial turnover.

    If Expert 2 ranks AAPL #10 today and #11 tomorrow, naive re-ranking
    sells AAPL and buys the new #10.  This 1.15x multiplier forces the
    algorithm to prefer holding existing positions unless a new asset
    genuinely dominates the incumbent.
    """
    buffered = raw_scores.copy()
    for ticker in current_positions:
        if ticker in buffered:
            buffered[ticker] *= buffer_multiplier
    return buffered


class SelectorPortfolioConstructor:
    def __init__(self, per_symbol_cap: float = 0.05):
        self.per_symbol_cap = per_symbol_cap
        self._current_holdings: list[str] = []

    def construct(
        self,
        scores: dict[str, float],
        current_positions: list[str] | None = None,
    ) -> list[dict]:
        if current_positions is not None:
            self._current_holdings = current_positions
        # Apply hysteresis before ranking
        buffered = apply_hysteresis_buffer(scores, self._current_holdings)
        weights = normalize_long_short(
            buffered, top_n=min(5, max(1, len(buffered) // 2)), gross_target=1.0,
        )
        capped = apply_caps(weights, self.per_symbol_cap)
        # Update holdings for next cycle
        self._current_holdings = [s for s, w in capped.items() if abs(w) > 0]
        return [{
            "symbol": s, "target_weight": round(w, 6),
        } for s, w in sorted(capped.items()) if abs(w) > 0]
