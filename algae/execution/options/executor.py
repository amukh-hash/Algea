from __future__ import annotations

from dataclasses import dataclass
from typing import List

from algae.execution.interfaces import ExecutionDecision


@dataclass
class PaperExecutor:
    decisions: List[ExecutionDecision]

    def execute(self, decision: ExecutionDecision) -> None:
        self.decisions.append(decision)


@dataclass
class NoopExecutor:
    def execute(self, _decision: ExecutionDecision) -> None:
        pass
