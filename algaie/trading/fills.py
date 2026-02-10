from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, List, Optional

import pandas as pd

from algaie.trading.orders import Fill, OrderIntent, Order
from algaie.trading.costs import SlippageModel


@dataclass(frozen=True)
class FillConfig:
    price_mode: str = "next_open"
    slippage_model: SlippageModel = SlippageModel()


class ExecutionSimulator:
    def __init__(self, config: FillConfig) -> None:
        self.config = config

    def simulate(
        self,
        intents: Iterable[OrderIntent],
        prices: pd.DataFrame,
        asof: date,
        next_date: Optional[date],
    ) -> tuple[List[Fill], List[Order]]:
        fills: List[Fill] = []
        orders: List[Order] = []
        for intent in intents:
            fill_date = asof
            if self.config.price_mode == "next_open":
                if next_date is None:
                    continue
                fill_date = next_date
                price = _price_for_ticker(prices, fill_date, intent.ticker, "open")
            elif self.config.price_mode == "close":
                price = _price_for_ticker(prices, asof, intent.ticker, "close")
            elif self.config.price_mode == "vwap":
                price = _price_for_ticker(prices, asof, intent.ticker, "vwap")
            else:
                raise ValueError(f"Unknown price_mode: {self.config.price_mode}")
            volume = _volume_for_ticker(prices, fill_date, intent.ticker)
            price = self.config.slippage_model.apply(price, intent.side, intent.quantity, volume)

            fills.append(
                Fill(asof=fill_date, ticker=intent.ticker, quantity=intent.quantity, price=price, side=intent.side)
            )
            orders.append(
                Order(
                    asof=intent.asof,
                    ticker=intent.ticker,
                    quantity=intent.quantity,
                    side=intent.side,
                    status="filled",
                    fill_price=price,
                    client_order_id=intent.client_order_id,
                )
            )
        return fills, orders


def _row_for_ticker(prices: pd.DataFrame, asof: date, ticker: str) -> pd.Series | None:
    """Return the first matching row for *ticker* on *asof*, or ``None``."""
    subset = prices[(prices["date"] == pd.Timestamp(asof)) & (prices["ticker"] == ticker)]
    if subset.empty:
        return None
    return subset.iloc[0]


def _price_for_ticker(prices: pd.DataFrame, asof: date, ticker: str, column: str) -> float:
    row = _row_for_ticker(prices, asof, ticker)
    if row is None:
        raise KeyError(f"Missing price for {ticker} on {asof}")
    if column not in row.index:
        raise KeyError(f"Missing column {column} for {ticker} on {asof}")
    return float(row[column])


def _volume_for_ticker(prices: pd.DataFrame, asof: date, ticker: str) -> float | None:
    row = _row_for_ticker(prices, asof, ticker)
    if row is None or "volume" not in row.index:
        return None
    return float(row["volume"])
