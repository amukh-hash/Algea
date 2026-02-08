from __future__ import annotations

from dataclasses import dataclass

from algaie.execution.interfaces import ExecutionContext, ExecutionDecision, SignalFrame
from algaie.execution.options.executor import NoopExecutor
from algaie.execution.options.gating import allow_trade
from algaie.execution.options.strategy import StrikeSelector


@dataclass
class OptionsStrategy:
    selector: StrikeSelector
    executor: NoopExecutor

    def run(self, signals: SignalFrame, context: ExecutionContext) -> list[ExecutionDecision]:
        decisions: list[ExecutionDecision] = []
        for _, row in signals.frame.iterrows():
            if not allow_trade(row["score"]):
                continue
            instrument = self.selector.select(row["ticker"])
            decision = ExecutionDecision(
                action="open",
                instrument=instrument,
                quantity=1.0,
                reason="score_positive",
            )
            self.executor.execute(decision)
            decisions.append(decision)
        return decisions
