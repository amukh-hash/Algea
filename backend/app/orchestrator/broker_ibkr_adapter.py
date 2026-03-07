"""Bridge between the existing IBKRLiveBroker and the orchestrator BrokerAdapter protocol.

This adapter lets the orchestrator talk to a *real* IBKR paper/live account
through the already-proven ``algae.trading.broker_ibkr.IBKRLiveBroker``.

Day-2 Mitigation — Broker Gateway Forced Disconnects:
    IBKR resets its API gateway daily (~23:45 EST).  All network-bound methods
    are wrapped with tenacity exponential backoff (4s→60s, 10 attempts, ~10 min
    window) to survive the reset without crashing the orchestrator.
"""
from __future__ import annotations

import logging
import os
from datetime import date, datetime
from typing import Any

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    before_sleep_log,
)

from algae.trading.broker_ibkr import IBKRLiveBroker, IbkrConfig
from algae.trading.orders import OrderIntent
from backend.app.schemas.fill_position import (
    FILLS_SCHEMA_VERSION,
    POSITIONS_SCHEMA_VERSION,
    normalize_fill,
    normalize_position,
)
from backend.app.contracts.validators import normalize_broker_positions_payload

logger = logging.getLogger(__name__)

# ── Retry policy for IBKR gateway disconnects ──────────────────────────
# Catches socket drops and TCP resets without swallowing logic errors.
# Total retry window: ~10 minutes (covers IBKR's daily gateway reboot).
_IBKR_RETRY = retry(
    retry=retry_if_exception_type((ConnectionError, TimeoutError, OSError)),
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(10),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class IBKRBrokerAdapter:
    """Wraps :class:`IBKRLiveBroker` to satisfy :class:`BrokerAdapter` protocol.

    All network-bound methods include exponential backoff retry logic
    to survive IBKR's daily gateway reset (~23:45 EST).
    """

    def __init__(self, broker: IBKRLiveBroker) -> None:
        self._broker = broker

    # -- Reconnection helper -----------------------------------------------------

    def _reconnect_if_needed(self) -> None:
        """Re-establish the IBKR connection if the session is stale.

        Called before each retried network operation to ensure the
        TCP socket is alive after a gateway reset.
        """
        try:
            if not self._broker._client.isConnected():
                logger.warning("IBKR session stale — reconnecting")
                self._broker._ensure_connected()
        except Exception:
            # Force full reconnect on any introspection failure
            logger.warning("IBKR connection check failed — forcing reconnect")
            try:
                self._broker._disconnect()
            except Exception:
                pass
            self._broker._ensure_connected()

    # -- BrokerAdapter protocol --------------------------------------------------

    def verify_paper(self) -> None:
        """Ensure the account is a paper account (DU prefix)."""
        self._broker._ensure_connected()
        account_id = self._broker.config.account_id
        if self._broker.config.paper_only and not account_id.startswith("DU"):
            raise RuntimeError(f"Paper guard: account '{account_id}' is not a paper account")
        logger.info("Paper guard passed for account %s", account_id[:4] + "****")

    @_IBKR_RETRY
    def place_orders(self, orders: dict) -> dict:
        """Convert orchestrator order dict → OrderIntent list → submit via IBKRLiveBroker."""
        self._reconnect_if_needed()

        order_list = orders.get("orders", [])
        if not order_list:
            return {"status": "accepted", "order_count": 0, "routed": []}

        asof = str(orders.get("asof_date", ""))
        intents = []
        for o in order_list:
            intents.append(
                OrderIntent(
                    asof=datetime.fromisoformat(asof).date() if asof else date(1970, 1, 1),
                    ticker=str(o["symbol"]),
                    quantity=float(o["qty"]),
                    side=str(o["side"]),
                    reason="orchestrator",
                    client_order_id=o.get("client_order_id"),
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

    @_IBKR_RETRY
    def get_positions(self) -> dict:
        """Return positions as a dict matching orchestrator expectations."""
        self._reconnect_if_needed()
        positions = self._broker.get_positions()
        payload = {
            "positions": [
                {
                    "symbol": p.ticker,
                    "quantity": p.quantity,
                    "avg_cost": p.avg_cost,
                }
                for p in positions
            ],
            "schema_version": POSITIONS_SCHEMA_VERSION,
        }
        return normalize_broker_positions_payload(payload, source="ibkr_adapter", compatibility_mode=True)

    @_IBKR_RETRY
    def get_fills(self, since_ts: str | None) -> dict:
        """Return fills as a dict matching orchestrator expectations."""
        self._reconnect_if_needed()

        time_min = None
        if since_ts:
            try:
                time_min = datetime.fromisoformat(since_ts)
            except ValueError:
                pass

        fills = self._broker.get_fills(time_min=time_min)
        normalized = [normalize_fill(
            {
                "ticker": f.ticker,
                "qty": f.quantity,
                "price": f.price,
                "side": f.side,
                "order_id": f.order_id,
                "commission": f.commission,
                "execution_time": str(f.execution_time) if f.execution_time else None,
            },
            source="ibkr",
        ).to_dict() for f in fills]
        return {"fills": normalized, "since": since_ts, "schema_version": FILLS_SCHEMA_VERSION}

    @_IBKR_RETRY
    def get_quote(self, symbol: str) -> float | None:
        """Fetch latest price for a symbol via IBKR historical data.

        Falls back to None if the symbol cannot be resolved.
        """
        self._reconnect_if_needed()

        try:
            from ib_insync import Stock  # type: ignore[import-untyped]

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
        except (ConnectionError, TimeoutError, OSError):
            raise  # Let tenacity handle these
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
