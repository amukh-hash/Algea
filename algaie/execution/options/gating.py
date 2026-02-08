from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateReasonCode:
    code: str
    message: str


def allow_trade(score: float) -> bool:
    return score > 0
