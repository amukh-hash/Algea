from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable, List, Optional

from algea.trading.orders import Fill, Order, OrderIntent


@dataclass(frozen=True)
class BrokerPosition:
    ticker: str
    quantity: float
    avg_cost: float


@dataclass(frozen=True)
class BrokerAccount:
    asof: date
    cash: float
    equity: float
    buying_power: float


class BrokerBase(ABC):
    @abstractmethod
    def submit_orders(self, intents: Iterable[OrderIntent]) -> List[Order]:
        raise NotImplementedError

    @abstractmethod
    def get_positions(self) -> List[BrokerPosition]:
        raise NotImplementedError

    @abstractmethod
    def get_orders(self, status: str | None = None) -> List[Order]:
        raise NotImplementedError

    @abstractmethod
    def get_account(self) -> BrokerAccount:
        raise NotImplementedError

    @abstractmethod
    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError

    def get_fills(
        self,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
    ) -> List[Fill]:
        """Retrieve fills/executions.  Default returns empty list."""
        return []
