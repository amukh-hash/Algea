from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Iterable, List, Optional

from algaie.trading.broker_base import BrokerAccount, BrokerBase, BrokerPosition
from algaie.trading.orders import Order, OrderIntent


@dataclass(frozen=True)
class IbkrConfig:
    gateway_url: str
    account_id: Optional[str] = None


class IBKRLiveBroker(BrokerBase):
    def __init__(self, config: IbkrConfig) -> None:
        self.config = config

    @classmethod
    def from_env(cls) -> "IBKRLiveBroker":
        gateway_url = os.getenv("IBKR_GATEWAY_URL")
        if not gateway_url:
            raise RuntimeError("Missing IBKR_GATEWAY_URL in environment")
        return cls(IbkrConfig(gateway_url=gateway_url, account_id=os.getenv("IBKR_ACCOUNT_ID")))

    def submit_orders(self, intents: Iterable[OrderIntent]) -> List[Order]:
        raise NotImplementedError("IBKRLiveBroker submit_orders not yet implemented")

    def get_positions(self) -> List[BrokerPosition]:
        raise NotImplementedError("IBKRLiveBroker get_positions not yet implemented")

    def get_orders(self, status: str | None = None) -> List[Order]:
        raise NotImplementedError("IBKRLiveBroker get_orders not yet implemented")

    def get_account(self) -> BrokerAccount:
        return BrokerAccount(asof=date.today(), cash=0.0, equity=0.0, buying_power=0.0)

    def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError("IBKRLiveBroker cancel_order not yet implemented")
