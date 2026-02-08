from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Iterable, List, Optional

import requests

from algaie.trading.broker_base import BrokerAccount, BrokerBase, BrokerPosition
from algaie.trading.orders import Order, OrderIntent


@dataclass(frozen=True)
class AlpacaConfig:
    base_url: str
    api_key: str
    secret_key: str


class AlpacaPaperBroker(BrokerBase):
    def __init__(self, config: AlpacaConfig) -> None:
        self.config = config
        self.session = requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": self.config.api_key,
                "APCA-API-SECRET-KEY": self.config.secret_key,
            }
        )

    @classmethod
    def from_env(cls) -> "AlpacaPaperBroker":
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        if not api_key or not secret_key:
            raise RuntimeError("Missing Alpaca credentials in environment")
        return cls(AlpacaConfig(base_url=base_url, api_key=api_key, secret_key=secret_key))

    def submit_orders(self, intents: Iterable[OrderIntent]) -> List[Order]:
        orders: List[Order] = []
        for intent in intents:
            payload = {
                "symbol": intent.ticker,
                "qty": str(intent.quantity),
                "side": intent.side,
                "type": "market",
                "time_in_force": "day",
                "client_order_id": intent.client_order_id,
            }
            response = self.session.post(f"{self.config.base_url}/v2/orders", json=payload, timeout=30)
            response.raise_for_status()
            data = response.json()
            orders.append(
                Order(
                    asof=intent.asof,
                    ticker=intent.ticker,
                    quantity=float(data.get("qty", intent.quantity)),
                    side=data.get("side", intent.side),
                    status=data.get("status", "submitted"),
                    broker_order_id=data.get("id"),
                    client_order_id=data.get("client_order_id"),
                )
            )
        return orders

    def get_positions(self) -> List[BrokerPosition]:
        response = self.session.get(f"{self.config.base_url}/v2/positions", timeout=30)
        response.raise_for_status()
        data = response.json()
        return [
            BrokerPosition(
                ticker=item["symbol"],
                quantity=float(item["qty"]),
                avg_cost=float(item.get("avg_entry_price", 0.0)),
            )
            for item in data
        ]

    def get_orders(self, status: Optional[str] = None) -> List[Order]:
        params = {"status": status} if status else {}
        response = self.session.get(f"{self.config.base_url}/v2/orders", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return [
            Order(
                asof=date.today(),
                ticker=item["symbol"],
                quantity=float(item.get("qty", 0.0)),
                side=item.get("side", "buy"),
                status=item.get("status", "unknown"),
                fill_price=float(item.get("filled_avg_price") or 0.0),
                broker_order_id=item.get("id"),
                client_order_id=item.get("client_order_id"),
            )
            for item in data
        ]

    def get_account(self) -> BrokerAccount:
        response = self.session.get(f"{self.config.base_url}/v2/account", timeout=30)
        response.raise_for_status()
        data = response.json()
        return BrokerAccount(
            asof=date.today(),
            cash=float(data.get("cash", 0.0)),
            equity=float(data.get("equity", 0.0)),
            buying_power=float(data.get("buying_power", 0.0)),
        )

    def cancel_order(self, order_id: str) -> None:
        response = self.session.delete(f"{self.config.base_url}/v2/orders/{order_id}", timeout=30)
        response.raise_for_status()
