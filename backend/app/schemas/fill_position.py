from __future__ import annotations

from dataclasses import dataclass
from typing import Any


FILLS_SCHEMA_VERSION = "fills.v1"
POSITIONS_SCHEMA_VERSION = "positions.v1"


@dataclass(frozen=True)
class FillRecord:
    schema_version: str
    symbol: str
    quantity: float
    price: float
    side: str
    filled_at: str | None
    source: str
    broker_order_id: str | None = None
    client_order_id: str | None = None
    commission: float = 0.0
    fill_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "price": self.price,
            "side": self.side,
            "filled_at": self.filled_at,
            "source": self.source,
            "broker_order_id": self.broker_order_id,
            "client_order_id": self.client_order_id,
            "commission": self.commission,
            "fill_id": self.fill_id,
        }


@dataclass(frozen=True)
class PositionRecord:
    schema_version: str
    symbol: str
    quantity: float
    avg_cost: float
    asof_date: str | None
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "quantity": self.quantity,
            "avg_cost": self.avg_cost,
            "asof_date": self.asof_date,
            "source": self.source,
        }


def normalize_fill(raw: dict[str, Any], source: str) -> FillRecord:
    symbol = str(raw.get("symbol") or raw.get("ticker") or "").strip()
    qty = float(raw.get("quantity", raw.get("qty", 0.0)))
    side = str(raw.get("side", "")).lower()
    return FillRecord(
        schema_version=FILLS_SCHEMA_VERSION,
        symbol=symbol,
        quantity=qty,
        price=float(raw.get("price", 0.0)),
        side=side,
        filled_at=raw.get("filled_at") or raw.get("ts") or raw.get("execution_time"),
        source=source,
        broker_order_id=raw.get("broker_order_id") or raw.get("order_id"),
        client_order_id=raw.get("client_order_id"),
        commission=float(raw.get("commission", 0.0)),
        fill_id=raw.get("fill_id") or raw.get("id"),
    )


def normalize_position(raw: dict[str, Any], source: str, asof_date: str | None = None) -> PositionRecord:
    symbol = str(raw.get("symbol") or raw.get("ticker") or "").strip()
    qty = float(raw.get("quantity", raw.get("qty", 0.0)))
    avg_cost = float(raw.get("avg_cost", raw.get("avgCost", 0.0)))
    return PositionRecord(
        schema_version=POSITIONS_SCHEMA_VERSION,
        symbol=symbol,
        quantity=qty,
        avg_cost=avg_cost,
        asof_date=asof_date,
        source=source,
    )


def validate_fills_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != FILLS_SCHEMA_VERSION:
        raise ValueError(f"invalid fills schema_version: {payload.get('schema_version')}")


def validate_positions_payload(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != POSITIONS_SCHEMA_VERSION:
        raise ValueError(f"invalid positions schema_version: {payload.get('schema_version')}")

