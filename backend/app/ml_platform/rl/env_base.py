from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StepResult:
    state: dict[str, float]
    reward: float
    done: bool
    info: dict


class TradingEnvBase:
    def __init__(self, seed: int = 7):
        self.seed = seed
        self.t = 0

    def reset(self) -> dict[str, float]:
        self.t = 0
        return self._state()

    def _state(self) -> dict[str, float]:
        raise NotImplementedError

    def step(self, action: dict) -> StepResult:
        raise NotImplementedError
