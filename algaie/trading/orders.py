from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass(frozen=True)
class OrderIntent:
    asof: date
    ticker: str
    quantity: float
    side: str
    reason: str
    tick_id: Optional[str] = None
    submitted_at: Optional[datetime] = None
    limit_price: Optional[float] = None
    client_order_id: Optional[str] = None


@dataclass(frozen=True)
class Order:
    asof: date
    ticker: str
    quantity: float
    side: str
    status: str
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    fill_price: Optional[float] = None
    broker_order_id: Optional[str] = None
    client_order_id: Optional[str] = None


@dataclass(frozen=True)
class Fill:
    asof: date
    ticker: str
    quantity: float
    price: float
    side: str
    order_id: Optional[str] = None
    commission: float = 0.0
    execution_time: Optional[datetime] = None
