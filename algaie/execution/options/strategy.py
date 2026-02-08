from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StrikeSelector:
    """Placeholder strategy for strike selection."""

    def select(self, ticker: str) -> str:
        return f"{ticker}-ATM"
