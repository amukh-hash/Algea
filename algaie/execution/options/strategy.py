from __future__ import annotations


class StrikeSelector:
    """Placeholder strategy for strike selection."""

    def select(self, ticker: str) -> str:
        return f"{ticker}-ATM"
