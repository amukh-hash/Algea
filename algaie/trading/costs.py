from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class CommissionModel:
    per_trade: float = 0.0
    per_share: float = 0.0
    bps: float = 0.0
    min_commission: float = 0.0

    def commission(self, shares: float, notional: float) -> float:
        value = self.per_trade + abs(shares) * self.per_share + abs(notional) * (self.bps / 10_000)
        return max(value, self.min_commission)


@dataclass(frozen=True)
class SlippageModel:
    model: Literal["none", "bps", "volume"] = "none"
    bps: float = 0.0
    volume_impact: float = 0.0

    def apply(self, price: float, side: str, qty: float, volume: float | None = None) -> float:
        if self.model == "none":
            return price
        if self.model == "bps":
            direction = 1 if side == "buy" else -1
            return price * (1 + direction * (self.bps / 10_000))
        if self.model == "volume":
            if not volume or volume <= 0:
                return price
            impact = self.volume_impact * np.sqrt(abs(qty) / volume)
            direction = 1 if side == "buy" else -1
            return price * (1 + direction * impact)
        raise ValueError(f"Unknown slippage model: {self.model}")
