"""Persistent paper broker — file-backed simulation of a real paper trading account.

Persists positions, cash, fills, and realized P&L to a JSON state file so that
state survives orchestrator and server restarts.  Orders are filled immediately
at the current market price (simulated market orders).
"""
from __future__ import annotations

import json
import logging
import os
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from backend.app.schemas.fill_position import (
    FILLS_SCHEMA_VERSION,
    POSITIONS_SCHEMA_VERSION,
    normalize_fill,
    normalize_position,
)

logger = logging.getLogger(__name__)

_DEFAULT_STATE_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "paper_account"


@dataclass
class _Position:
    symbol: str
    qty: float
    avg_cost: float  # per-unit cost in index points (NOT notional)

    def to_dict(self) -> dict[str, Any]:
        return {"symbol": self.symbol, "qty": self.qty, "avg_cost": round(self.avg_cost, 8)}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "_Position":
        return cls(symbol=str(d["symbol"]), qty=float(d["qty"]), avg_cost=float(d["avg_cost"]))


@dataclass
class _Fill:
    fill_id: str
    ts: str
    symbol: str
    qty: float
    price: float
    side: str
    commission: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.fill_id,
            "ts": self.ts,
            "symbol": self.symbol,
            "qty": self.qty,
            "price": round(self.price, 8),
            "side": self.side,
            "commission": self.commission,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "_Fill":
        return cls(
            fill_id=str(d.get("id", "")),
            ts=str(d.get("ts", "")),
            symbol=str(d["symbol"]),
            qty=float(d["qty"]),
            price=float(d["price"]),
            side=str(d["side"]),
            commission=float(d.get("commission", 0.0)),
        )


class PersistentPaperBroker:
    """File-backed paper broker that simulates a real trading account.

    All state is persisted to ``state_dir/state.json``.  Thread-safe via a
    reentrant lock so it can be used safely from the orchestrator daemon.
    """

    def __init__(
        self,
        state_dir: Path | None = None,
        starting_equity: float = 100_000.0,
        account_id: str = "PAPER001",
    ) -> None:
        self.state_dir = Path(state_dir or os.getenv("PAPER_STATE_DIR", str(_DEFAULT_STATE_DIR)))
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._state_path = self.state_dir / "state.json"
        self._lock = threading.RLock()
        self.account_id = account_id
        self.is_paper = True

        # Load or initialize state
        if self._state_path.exists():
            self._load()
            logger.info(
                "Loaded paper account: cash=%.2f, positions=%d, fills=%d",
                self._cash,
                len(self._positions),
                len(self._fills),
            )
        else:
            self._starting_equity = starting_equity
            self._cash = starting_equity
            self._positions: dict[str, _Position] = {}
            self._fills: list[_Fill] = []
            self._realized_pnl = 0.0
            self._save()
            logger.info("Initialized new paper account: equity=%.2f", starting_equity)

        # Price cache for simulated fills — populated via get_quote or external injection
        self._price_cache: dict[str, float] = {}

    # ── BrokerAdapter protocol ──────────────────────────────────────────

    def verify_paper(self) -> None:
        if not self.is_paper:
            raise RuntimeError("Paper guard: broker flagged non-paper")

    def place_orders(self, orders: dict) -> dict:
        """Simulate immediate fills for all orders at current market price."""
        order_list = orders.get("orders", [])
        if not order_list:
            return {"status": "accepted", "order_count": 0, "account": self.account_id, "routed": []}

        routed = []
        with self._lock:
            for o in order_list:
                symbol = str(o["symbol"])
                qty = float(o["qty"])
                side = str(o["side"]).upper()
                fill_price = self._resolve_price(symbol, o)

                if fill_price is None:
                    logger.warning("No price for %s, skipping order", symbol)
                    routed.append({
                        "ticker": symbol, "qty": qty, "side": side,
                        "status": "rejected", "reason": "no_price",
                    })
                    continue

                # Execute the fill
                signed_qty = qty if side == "BUY" else -qty
                self._apply_fill(symbol, signed_qty, fill_price)

                fill = _Fill(
                    fill_id=str(uuid.uuid4())[:8],
                    ts=datetime.now(timezone.utc).isoformat(),
                    symbol=symbol,
                    qty=qty,
                    price=fill_price,
                    side=side,
                )
                self._fills.append(fill)

                routed.append({
                    "ticker": symbol, "qty": qty, "side": side,
                    "status": "filled", "fill_price": fill_price,
                    "broker_order_id": fill.fill_id,
                })
                logger.info(
                    "Paper fill: %s %s %.0f @ %.4f",
                    side, symbol, qty, fill_price,
                )

            self._save()

        return {
            "status": "accepted",
            "order_count": len(routed),
            "account": self.account_id,
            "routed": routed,
        }

    def get_positions(self) -> dict:
        with self._lock:
            positions = [
                normalize_position(p.to_dict(), source="paper").to_dict()
                for p in self._positions.values() if p.qty != 0
            ]
            return {
                "schema_version": POSITIONS_SCHEMA_VERSION,
                "positions": positions,
            }

    def get_fills(self, since_ts: str | None) -> dict:
        with self._lock:
            fills = self._fills
            if since_ts:
                fills = [f for f in fills if f.ts >= since_ts]
            normalized = [normalize_fill(f.to_dict(), source="paper").to_dict() for f in fills]
            return {"schema_version": FILLS_SCHEMA_VERSION, "fills": normalized, "since": since_ts}

    def get_quote(self, symbol: str) -> float | None:
        """Return cached price or fetch live via yfinance."""
        cached = self._price_cache.get(symbol)
        if cached is not None:
            return cached

        # Try to fetch live
        try:
            from backend.app.api.live_prices import get_live_prices
            prices = get_live_prices([symbol])
            if symbol in prices:
                self._price_cache[symbol] = prices[symbol]
                return prices[symbol]
        except Exception as exc:
            logger.debug("get_quote(%s) live fetch failed: %s", symbol, exc)

        return None

    # ── Account state accessors ─────────────────────────────────────────

    @property
    def cash(self) -> float:
        return self._cash

    @property
    def starting_equity(self) -> float:
        return self._starting_equity

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    def get_account_summary(self) -> dict[str, Any]:
        with self._lock:
            return {
                "account_id": self.account_id,
                "starting_equity": self._starting_equity,
                "cash": round(self._cash, 2),
                "position_count": len([p for p in self._positions.values() if p.qty != 0]),
                "realized_pnl": round(self._realized_pnl, 2),
                "fill_count": len(self._fills),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

    # ── Internal helpers ────────────────────────────────────────────────

    def _resolve_price(self, symbol: str, order: dict) -> float | None:
        """Find a price: est_price from order > cache > live fetch."""
        # Use order's estimated price if available
        est = order.get("est_price")
        if est and float(est) > 0:
            self._price_cache[symbol] = float(est)
            return float(est)
        # Try cache
        if symbol in self._price_cache:
            return self._price_cache[symbol]
        # Try live
        return self.get_quote(symbol)

    def _apply_fill(self, symbol: str, signed_qty: float, price: float) -> None:
        """Update position and cash for a fill.  signed_qty is positive for buy."""
        pos = self._positions.get(symbol)

        if pos is None:
            # New position
            self._positions[symbol] = _Position(symbol=symbol, qty=signed_qty, avg_cost=price)
            # Deduct cash (buy costs money, sell receives)
            self._cash -= signed_qty * price
        else:
            old_qty = pos.qty
            new_qty = old_qty + signed_qty

            if old_qty == 0:
                # Was flat, opening fresh
                pos.qty = new_qty
                pos.avg_cost = price
                self._cash -= signed_qty * price
            elif (old_qty > 0 and signed_qty > 0) or (old_qty < 0 and signed_qty < 0):
                # Adding to position — average the cost
                total_cost = pos.avg_cost * abs(old_qty) + price * abs(signed_qty)
                pos.qty = new_qty
                pos.avg_cost = total_cost / abs(new_qty)
                self._cash -= signed_qty * price
            else:
                # Reducing or flipping position — realize P&L on closed portion
                closed_qty = min(abs(signed_qty), abs(old_qty))
                if old_qty > 0:
                    # Was long, selling
                    realized = (price - pos.avg_cost) * closed_qty
                else:
                    # Was short, buying to cover
                    realized = (pos.avg_cost - price) * closed_qty
                self._realized_pnl += realized
                self._cash -= signed_qty * price

                if abs(new_qty) < 1e-10:
                    # Position closed
                    pos.qty = 0
                elif (new_qty > 0) != (old_qty > 0):
                    # Position flipped — remaining qty at new price
                    pos.qty = new_qty
                    pos.avg_cost = price
                else:
                    # Partially closed — avg_cost stays the same
                    pos.qty = new_qty

    def _load(self) -> None:
        data = json.loads(self._state_path.read_text(encoding="utf-8-sig"))
        self._starting_equity = float(data.get("starting_equity", 100_000))
        self._cash = float(data.get("cash", self._starting_equity))
        self._positions = {
            p["symbol"]: _Position.from_dict(p) for p in data.get("positions", [])
        }
        self._fills = [_Fill.from_dict(f) for f in data.get("fills", [])]
        self._realized_pnl = float(data.get("realized_pnl", 0.0))

    def _save(self) -> None:
        data = {
            "account_id": self.account_id,
            "starting_equity": self._starting_equity,
            "cash": round(self._cash, 2),
            "positions": [p.to_dict() for p in self._positions.values() if p.qty != 0],
            "fills": [f.to_dict() for f in self._fills],
            "realized_pnl": round(self._realized_pnl, 2),
            "last_updated": datetime.now(timezone.utc).isoformat(),
        }
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self._state_path)
