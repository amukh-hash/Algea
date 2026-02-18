"""Bridge between the existing IBKRLiveBroker and the orchestrator BrokerAdapter protocol.

This adapter lets the orchestrator talk to a *real* IBKR paper/live account
through the already-proven ``algaie.trading.broker_ibkr.IBKRLiveBroker``.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any

from algaie.trading.broker_ibkr import IBKRLiveBroker, IbkrConfig
from algaie.trading.orders import OrderIntent

logger = logging.getLogger(__name__)


class IBKRBrokerAdapter:
    """Wraps :class:`IBKRLiveBroker` to satisfy :class:`BrokerAdapter` protocol."""

    def __init__(self, broker: IBKRLiveBroker) -> None:
        self._broker = broker

    # -- BrokerAdapter protocol --------------------------------------------------

    def verify_paper(self) -> None:
        """Ensure the account is a paper account (DU prefix)."""
        self._broker._ensure_connected()
        account_id = self._broker.config.account_id
        if self._broker.config.paper_only and not account_id.startswith("DU"):
            raise RuntimeError(f"Paper guard: account '{account_id}' is not a paper account")
        logger.info("Paper guard passed for account %s", account_id[:4] + "****")

    def place_orders(self, orders: dict) -> dict:
        """Convert orchestrator order dict → OrderIntent list → submit via IBKRLiveBroker."""
        order_list = orders.get("orders", [])
        if not order_list:
            return {"status": "accepted", "order_count": 0, "routed": []}

        today = date.today()
        intents = []
        for o in order_list:
            intents.append(
                OrderIntent(
                    asof=today,
                    ticker=str(o["symbol"]),
                    quantity=float(o["qty"]),
                    side=str(o["side"]),
                    reason="orchestrator",
                )
            )

        submitted = self._broker.submit_orders(intents)

        routed = []
        for s in submitted:
            routed.append({
                "ticker": s.ticker,
                "qty": s.quantity,
                "side": s.side,
                "status": s.status,
                "broker_order_id": getattr(s, "broker_order_id", None),
            })

        return {
            "status": "accepted",
            "order_count": len(routed),
            "account": self._broker.config.account_id,
            "routed": routed,
        }

    def get_positions(self) -> dict:
        """Return positions as a dict matching orchestrator expectations."""
        positions = self._broker.get_positions()
        return {
            "positions": [
                {
                    "symbol": p.ticker,
                    "qty": p.quantity,
                    "avg_cost": p.avg_cost,
                }
                for p in positions
            ]
        }

    def get_fills(self, since_ts: str | None) -> dict:
        """Return fills as a dict matching orchestrator expectations."""
        time_min = None
        if since_ts:
            try:
                time_min = datetime.fromisoformat(since_ts)
            except ValueError:
                pass

        fills = self._broker.get_fills(time_min=time_min)
        return {
            "fills": [
                {
                    "ticker": f.ticker,
                    "qty": f.quantity,
                    "price": f.price,
                    "side": f.side,
                    "order_id": f.order_id,
                    "commission": f.commission,
                }
                for f in fills
            ],
            "since": since_ts,
        }

    def get_quote(self, symbol: str) -> float | None:
        """Fetch latest price for a symbol via IBKR historical data.

        Falls back to None if the symbol cannot be resolved.
        """
        try:
            from ib_insync import Stock  # type: ignore[import-untyped]

            self._broker._ensure_connected()
            contract = Stock(symbol, "SMART", "USD")
            qualified = self._broker._client.qualify_contracts(contract)
            if not qualified or qualified[0].conId == 0:
                return None
            bars = self._broker._client.historical_bars(
                qualified[0], duration="1 D", bar_size="1 day"
            )
            if bars.empty:
                return None
            return float(bars.iloc[-1]["close"])
        except Exception as exc:
            logger.warning("get_quote(%s) failed: %s", symbol, exc)
            return None

    # -- Factory -----------------------------------------------------------------

    @classmethod
    def from_env(cls) -> "IBKRBrokerAdapter":
        """Create adapter from standard IBKR_* environment variables.

        Reads the same env vars as ``IBKRLiveBroker.from_env()``.
        """
        broker = IBKRLiveBroker.from_env()
        return cls(broker)

    # -- Lifecycle ---------------------------------------------------------------

    def disconnect(self) -> None:
        """Disconnect the underlying broker."""
        self._broker._disconnect()
