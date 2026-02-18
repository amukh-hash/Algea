from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Protocol


class BrokerAdapter(Protocol):
    def verify_paper(self) -> None:
        ...

    def place_orders(self, orders: dict) -> dict:
        ...

    def get_positions(self) -> dict:
        ...

    def get_fills(self, since_ts: str | None) -> dict:
        ...

    def get_quote(self, symbol: str) -> float | None:
        """Return latest price for *symbol*, or None if unavailable."""
        ...


@dataclass
class PaperBrokerStub:
    account_id: str = "DU999999"
    is_paper: bool = True
    # Optional pre-loaded price map for testing / daily-close injection
    price_map: dict[str, float] = field(default_factory=dict)

    def verify_paper(self) -> None:
        if not self.is_paper:
            raise RuntimeError("Paper guard: broker flagged non-paper")
        if os.getenv("ORCH_PAPER_ONLY", "1") == "1" and not self.account_id.startswith("DU"):
            raise RuntimeError(f"Paper guard: account '{self.account_id}' is not a paper account")

    def place_orders(self, orders: dict) -> dict:
        return {"status": "accepted", "order_count": len(orders.get("orders", [])), "account": self.account_id}

    def get_positions(self) -> dict:
        return {"positions": []}

    def get_fills(self, since_ts: str | None) -> dict:
        return {"fills": [], "since": since_ts}

    def get_quote(self, symbol: str) -> float | None:
        return self.price_map.get(symbol)
