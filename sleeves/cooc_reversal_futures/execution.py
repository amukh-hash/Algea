from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class FuturesOrder:
    symbol: str
    qty: int
    side: str
    order_type: str = "MKT_WITH_PROTECTION"


def build_entry_orders(contracts: dict[str, int]) -> list[FuturesOrder]:
    orders = []
    for symbol, qty in contracts.items():
        if qty == 0:
            continue
        orders.append(FuturesOrder(symbol=symbol, qty=abs(qty), side="BUY" if qty > 0 else "SELL"))
    return orders


def flatten_orders(positions: dict[str, int]) -> list[FuturesOrder]:
    return [FuturesOrder(symbol=s, qty=abs(q), side="SELL" if q > 0 else "BUY") for s, q in positions.items() if q != 0]


def emergency_flatten_needed(now_et: datetime, cutoff_et: datetime, positions: dict[str, int]) -> bool:
    return now_et >= cutoff_et and any(v != 0 for v in positions.values())
