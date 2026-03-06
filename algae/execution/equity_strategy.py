from __future__ import annotations

from dataclasses import dataclass

from algae.execution.interfaces import ExecutionDecision, SignalFrame

# Named constants for decision fields
ACTION_OPEN = "open"
REASON_SCORE_RANKED = "score_ranked"
DEFAULT_QUANTITY = 1.0


@dataclass
class EquityStrategy:
    def run(self, signals: SignalFrame) -> list[ExecutionDecision]:
        tickers = signals.frame["ticker"]
        return [
            ExecutionDecision(
                action=ACTION_OPEN,
                instrument=ticker,
                quantity=DEFAULT_QUANTITY,
                reason=REASON_SCORE_RANKED,
            )
            for ticker in tickers
        ]
