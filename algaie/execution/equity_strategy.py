from __future__ import annotations

from dataclasses import dataclass

from algaie.execution.interfaces import ExecutionDecision, SignalFrame


@dataclass
class EquityStrategy:
    def run(self, signals: SignalFrame) -> list[ExecutionDecision]:
        decisions: list[ExecutionDecision] = []
        for _, row in signals.frame.iterrows():
            decision = ExecutionDecision(
                action="open",
                instrument=row["ticker"],
                quantity=1.0,
                reason="score_ranked",
            )
            decisions.append(decision)
        return decisions
