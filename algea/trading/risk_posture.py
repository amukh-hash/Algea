"""
Risk posture — market-regime-aware position sizing and crash override.

Ported from deprecated/backend_app_snapshot/risk/risk_manager.py.
Decoupled from internal type imports (enums defined inline).
"""
from __future__ import annotations

import enum
from dataclasses import dataclass
from typing import Dict


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class RiskPosture(enum.Enum):
    NORMAL = "normal"
    CAUTIOUS = "cautious"
    DEFENSIVE = "defensive"


class ActionType(enum.Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    LIQUIDATE = "liquidate"
    NO_NEW_RISK = "no_new_risk"


@dataclass
class RiskDecision:
    symbol: str
    action: ActionType
    quantity: float
    reason: str


# ---------------------------------------------------------------------------
# Crash override (BPI + AD slope)
# ---------------------------------------------------------------------------

class CrashOverride:
    """
    Determines market-level risk posture from breadth indicators.

    Parameters
    ----------
    bpi_caution : BPI below this → cautious
    bpi_defensive : BPI below this → defensive
    ad_slope_caution : negative AD slope below this → cautious bump
    """

    def __init__(
        self,
        bpi_caution: float = 40.0,
        bpi_defensive: float = 25.0,
        ad_slope_caution: float = -0.5,
    ) -> None:
        self.bpi_caution = bpi_caution
        self.bpi_defensive = bpi_defensive
        self.ad_slope_caution = ad_slope_caution

    def check(self, bpi: float, ad_slope: float = 0.0) -> RiskPosture:
        if bpi < self.bpi_defensive:
            return RiskPosture.DEFENSIVE
        if bpi < self.bpi_caution or ad_slope < self.ad_slope_caution:
            return RiskPosture.CAUTIOUS
        return RiskPosture.NORMAL


# ---------------------------------------------------------------------------
# Risk manager
# ---------------------------------------------------------------------------

class RiskManager:
    """
    Per-symbol risk decisioning using model signal + breadth data.

    Usage::

        rm = RiskManager()
        decision = rm.evaluate(
            symbol="AAPL",
            prob_up=0.72,
            qty_held=0,
            breadth_data={"bpi": 55.0, "ad_slope": 0.1},
        )
    """

    def __init__(self) -> None:
        self.crash_override = CrashOverride()
        self._posture_handlers = {
            RiskPosture.DEFENSIVE: self._evaluate_defensive,
            RiskPosture.CAUTIOUS: self._evaluate_cautious,
            RiskPosture.NORMAL: self._evaluate_normal,
        }

    def _hold_or_no_risk(self, symbol: str, qty_held: int, reason: str) -> RiskDecision:
        action = ActionType.HOLD if qty_held > 0 else ActionType.NO_NEW_RISK
        return RiskDecision(symbol, action, 0, reason)

    def _evaluate_defensive(
        self, symbol: str, prob_up: float, qty_held: int, reason: str, **kwargs: float
    ) -> RiskDecision:
        if qty_held > 0 and prob_up < 0.4:
            return RiskDecision(symbol, ActionType.LIQUIDATE, -qty_held, reason + " | Defensive+Bearish")
        return self._hold_or_no_risk(symbol, qty_held, reason)

    def _evaluate_cautious(
        self, symbol: str, prob_up: float, qty_held: int, reason: str, **kwargs: float
    ) -> RiskDecision:
        if qty_held > 0 and prob_up < 0.4:
            return RiskDecision(symbol, ActionType.SELL, -qty_held, reason)
        if qty_held == 0 and prob_up > 0.7:
            return RiskDecision(symbol, ActionType.BUY, kwargs.get("cautious_buy_qty", 10.0), reason + " | Cautious entry")
        return self._hold_or_no_risk(symbol, qty_held, reason)

    def _evaluate_normal(
        self, symbol: str, prob_up: float, qty_held: int, reason: str, **kwargs: float
    ) -> RiskDecision:
        if qty_held > 0 and prob_up < 0.4:
            return RiskDecision(symbol, ActionType.SELL, -qty_held, reason)
        if qty_held == 0 and prob_up > 0.6:
            return RiskDecision(symbol, ActionType.BUY, kwargs.get("buy_qty", 20.0), reason)
        return self._hold_or_no_risk(symbol, qty_held, reason)

    def evaluate(
        self,
        symbol: str,
        prob_up: float,
        qty_held: int = 0,
        breadth_data: Dict[str, float] | None = None,
        *,
        buy_qty: float = 20.0,
        cautious_buy_qty: float = 10.0,
    ) -> RiskDecision:
        breadth = breadth_data or {}
        bpi = breadth.get("bpi", 50.0)
        ad_slope = breadth.get("ad_slope", 0.0)

        posture = self.crash_override.check(bpi, ad_slope)
        reason = f"Posture: {posture.value}, P(up)={prob_up:.2f}"

        handler = self._posture_handlers[posture]
        return handler(symbol, prob_up, qty_held, reason, buy_qty=buy_qty, cautious_buy_qty=cautious_buy_qty)
