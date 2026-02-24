from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List

import pandas as pd

from algaie.data.common import ensure_datetime
from algaie.trading.orders import Fill, OrderIntent


@dataclass
class Position:
    ticker: str
    quantity: float
    avg_cost: float
    entry_date: date
    realized_pnl: float = 0.0


@dataclass
class PortfolioSnapshot:
    asof: date
    equity: float
    cash: float
    gross_exposure: float
    net_exposure: float


@dataclass
class Portfolio:
    cash: float
    positions: Dict[str, Position] = field(default_factory=dict)
    trade_log: List[dict] = field(default_factory=list)

    def total_equity(self, prices: pd.DataFrame, asof: date) -> float:
        mtm = 0.0
        for position in self.positions.values():
            price = _price_for_ticker(prices, asof, position.ticker)
            mtm += position.quantity * price
        return self.cash + mtm

    def snapshot(self, prices: pd.DataFrame, asof: date) -> PortfolioSnapshot:
        equity = self.total_equity(prices, asof)
        gross = 0.0
        net = 0.0
        for position in self.positions.values():
            price = _price_for_ticker(prices, asof, position.ticker)
            exposure = position.quantity * price
            gross += abs(exposure)
            net += exposure
        return PortfolioSnapshot(asof=asof, equity=equity, cash=self.cash, gross_exposure=gross, net_exposure=net)

    def _record_trade(self, position: Position, exit_date: date, exit_px: float, reason: str) -> float:
        """Log a closed trade and return the realized PnL."""
        realized = (exit_px - position.avg_cost) * position.quantity
        position.realized_pnl += realized
        denom = abs(position.avg_cost * position.quantity) if position.quantity else 0.0
        self.trade_log.append(
            {
                "entry_date": position.entry_date,
                "exit_date": exit_date,
                "ticker": position.ticker,
                "qty": position.quantity,
                "entry_px": position.avg_cost,
                "exit_px": exit_px,
                "pnl": realized,
                "ret": realized / denom if denom else 0.0,
                "hold_days": (exit_date - position.entry_date).days,
                "reason": reason,
            }
        )
        return realized

    def update_from_fill(self, fill: Fill, asof: date) -> None:
        signed_qty = fill.quantity if fill.side == "buy" else -fill.quantity
        position = self.positions.get(fill.ticker)
        if position is None:
            if signed_qty == 0:
                return
            self.positions[fill.ticker] = Position(
                ticker=fill.ticker,
                quantity=signed_qty,
                avg_cost=fill.price,
                entry_date=asof,
            )
            self.cash -= signed_qty * fill.price
            return

        old_qty = position.quantity
        new_qty = old_qty + signed_qty
        if old_qty == 0:
            position.avg_cost = fill.price
            position.entry_date = asof
        if new_qty == 0:
            self._record_trade(position, asof, fill.price, "exit")
            self.cash += position.quantity * fill.price
            self.positions.pop(fill.ticker, None)
            return

        # Flip side: close old, open new remainder at fill price
        if (old_qty > 0 > new_qty) or (old_qty < 0 < new_qty):
            self._record_trade(position, asof, fill.price, "flip")
            position.quantity = new_qty
            position.avg_cost = fill.price
            position.entry_date = asof

        # Same side increase: weighted average cost update
        elif (old_qty > 0 and signed_qty > 0) or (old_qty < 0 and signed_qty < 0):
            total_cost = position.avg_cost * position.quantity + fill.price * signed_qty
            position.quantity = new_qty
            position.avg_cost = total_cost / position.quantity

        # Partial reduction: realize proportional pnl; keep remaining avg_cost unchanged
        else:
            reduced_qty = min(abs(old_qty), abs(signed_qty))
            pnl = (fill.price - position.avg_cost) * (reduced_qty if old_qty > 0 else -reduced_qty)
            position.realized_pnl += pnl
            self.trade_log.append(
                {
                    "entry_date": position.entry_date,
                    "exit_date": asof,
                    "ticker": position.ticker,
                    "qty": reduced_qty if old_qty > 0 else -reduced_qty,
                    "entry_px": position.avg_cost,
                    "exit_px": fill.price,
                    "pnl": pnl,
                    "ret": pnl / (abs(position.avg_cost * reduced_qty) if reduced_qty else 1.0),
                    "hold_days": (asof - position.entry_date).days,
                    "reason": "partial_exit",
                }
            )
            position.quantity = new_qty
        self.cash -= signed_qty * fill.price

    def build_order_intents(
        self,
        asof: date,
        target_weights: pd.DataFrame,
        prices: pd.DataFrame,
        equity: float,
        rounding_policy: str,
    ) -> List[OrderIntent]:
        intents: List[OrderIntent] = []
        target_map = target_weights.set_index("ticker")["target_weight"].to_dict()
        tickers = set(target_map) | set(self.positions)
        for ticker in sorted(tickers):
            price = _price_for_ticker(prices, asof, ticker)
            target_weight = target_map.get(ticker, 0.0)
            target_value = equity * target_weight
            target_qty = target_value / price if price > 0 else 0.0
            if rounding_policy == "round":
                target_qty = float(round(target_qty))
            current_pos = self.positions.get(ticker)
            current_qty = current_pos.quantity if current_pos else 0.0
            delta = target_qty - current_qty
            if abs(delta) < 1e-8:
                continue
            side = "buy" if delta > 0 else "sell"
            intents.append(
                OrderIntent(asof=asof, ticker=ticker, quantity=abs(delta), side=side, reason="rebalance")
            )
        return intents


def _price_for_ticker(prices: pd.DataFrame, asof: date, ticker: str) -> float:
    ensure_datetime(prices, "date")
    subset = prices[(prices["date"] == pd.Timestamp(asof)) & (prices["ticker"] == ticker)]
    if subset.empty:
        raise KeyError(f"Missing price for {ticker} on {asof}")
    return float(subset.iloc[0]["close"])
