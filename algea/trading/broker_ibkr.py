"""IBKR live/paper broker — full implementation.

Uses :class:`IbkrClient` for isolation from ``ib_insync`` internals.
Supports dependency injection of ``client`` for mock-based testing.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

from ib_insync import LimitOrder, MarketOrder, Order as IbOrder  # type: ignore[import-untyped]

from algea.trading.broker_base import BrokerAccount, BrokerBase, BrokerPosition
from algea.trading.orders import Fill, Order, OrderIntent
from algea.trading.ibkr_client import IbkrClient
from algea.trading.ibkr_contracts import (
    build_future_contract,
    parse_active_contract_symbol,
    qualify_future,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _mask_account(account_id: str) -> str:
    """Mask account id for logging: ``DUP905542I`` → ``DUP*****42I``."""
    if len(account_id) <= 5:
        return "***"
    return account_id[:3] + "*" * (len(account_id) - 5) + account_id[-2:]


@dataclass(frozen=True)
class IbkrConfig:
    """Configuration for IBKR broker connection."""

    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 17
    account_id: str = ""
    paper_only: bool = True
    readonly: bool = False
    timeout: int = 20
    max_retries: int = 3


# ---------------------------------------------------------------------------
# Broker implementation
# ---------------------------------------------------------------------------


class IBKRLiveBroker(BrokerBase):
    """Full IBKR broker implementation for futures paper trading.

    Parameters
    ----------
    config
        Broker configuration.
    client
        Optional :class:`IbkrClient` for dependency injection (mock testing).
        If ``None``, a real client is created from *config*.
    """

    def __init__(
        self,
        config: IbkrConfig,
        client: Optional[IbkrClient] = None,
    ) -> None:
        if not config.host:
            raise ValueError("IbkrConfig.host must not be empty")
        self.config = config
        self._client = client or IbkrClient(
            host=config.host,
            port=config.port,
            client_id=config.client_id,
            timeout=config.timeout,
            readonly=config.readonly,
            max_retries=config.max_retries,
        )
        # Track broker_order_id → ib_insync Trade for cancel/status
        self._trade_map: Dict[int, Any] = {}

        if config.account_id:
            logger.info(
                "IBKRLiveBroker initialized: account=%s, paper=%s, readonly=%s",
                _mask_account(config.account_id),
                config.paper_only,
                config.readonly,
            )

    @classmethod
    def from_env(cls) -> "IBKRLiveBroker":
        """Create broker from environment variables.

        Reads
        -----
        IBKR_GATEWAY_URL : str  (``host:port`` or ``host``)
        IBKR_ACCOUNT_ID : str
        IBKR_CLIENT_ID : str  (optional, default ``17``)
        IBKR_PAPER_ONLY : str (optional, ``"1"`` = enforce paper checks)
        IBKR_READONLY : str   (optional, ``"1"`` = block submissions)
        """
        gateway_url = os.getenv("IBKR_GATEWAY_URL", "127.0.0.1:7497")
        account_id = os.getenv("IBKR_ACCOUNT_ID", "")

        # Parse host:port
        if ":" in gateway_url:
            parts = gateway_url.rsplit(":", 1)
            host = parts[0]
            port = int(parts[1])
        else:
            host = gateway_url
            port = 7497

        client_id = int(os.getenv("IBKR_CLIENT_ID", "17"))
        paper_only = os.getenv("IBKR_PAPER_ONLY", "1") == "1"
        readonly = os.getenv("IBKR_READONLY", "0") == "1"

        missing: List[str] = []
        if not host:
            missing.append("IBKR_GATEWAY_URL")
        if missing:
            raise RuntimeError(f"Missing required env vars: {', '.join(missing)}")

        config = IbkrConfig(
            host=host,
            port=port,
            client_id=client_id,
            account_id=account_id,
            paper_only=paper_only,
            readonly=readonly,
        )
        return cls(config)

    # -- connection ---------------------------------------------------------

    def _connect(self) -> None:
        self._client.connect()

    def _disconnect(self) -> None:
        self._client.disconnect()

    def _ensure_connected(self) -> None:
        self._client.ensure_connected()

    # -- submit_orders ------------------------------------------------------

    def submit_orders(self, intents: Iterable[OrderIntent]) -> List[Order]:
        """Submit order intents to IBKR.

        Intents are sorted deterministically by ``(ticker, side, quantity)``
        before submission.  If ``readonly=True`` in config, raises
        :class:`PermissionError`.
        """
        if self.config.readonly:
            raise PermissionError(
                "IBKRLiveBroker is in readonly mode — order submission blocked. "
                "Set IBKR_READONLY=0 to enable."
            )

        self._ensure_connected()
        intent_list = sorted(intents, key=lambda i: (i.ticker, i.side, i.quantity))
        orders: List[Order] = []

        for intent in intent_list:
            try:
                # Parse the intent ticker as an active contract symbol
                root, expiry = parse_active_contract_symbol(intent.ticker)
                contract = build_future_contract(root, expiry)

                # Qualify
                qualified = self._client.qualify_contracts(contract)
                if not qualified or qualified[0].conId == 0:
                    logger.error("Failed to qualify contract for %s", intent.ticker)
                    orders.append(
                        Order(
                            asof=intent.asof,
                            ticker=intent.ticker,
                            quantity=intent.quantity,
                            side=intent.side,
                            status="rejected",
                            client_order_id=intent.client_order_id,
                        )
                    )
                    continue

                ib_contract = qualified[0]

                # Build IBKR order
                action = "BUY" if intent.side.upper() in ("BUY", "B") else "SELL"
                qty = abs(int(intent.quantity))

                if intent.limit_price is not None:
                    ib_order = LimitOrder(action, qty, intent.limit_price)
                else:
                    ib_order = MarketOrder(action, qty)

                # Set account
                if self.config.account_id:
                    ib_order.account = self.config.account_id

                # Place
                trade = self._client.place_order(ib_contract, ib_order)
                self._client.sleep(0.3)  # allow async event processing

                broker_id = str(trade.order.orderId)
                self._trade_map[trade.order.orderId] = trade

                orders.append(
                    Order(
                        asof=intent.asof,
                        ticker=intent.ticker,
                        quantity=intent.quantity,
                        side=intent.side,
                        status=trade.orderStatus.status if trade.orderStatus else "submitted",
                        broker_order_id=broker_id,
                        client_order_id=intent.client_order_id,
                    )
                )
                logger.info(
                    "Order placed: %s %s %d %s → broker_id=%s status=%s",
                    action,
                    intent.ticker,
                    qty,
                    ib_contract.localSymbol,
                    broker_id,
                    trade.orderStatus.status if trade.orderStatus else "submitted",
                )

            except Exception as exc:
                logger.error("Order submission failed for %s: %s", intent.ticker, exc)
                orders.append(
                    Order(
                        asof=intent.asof,
                        ticker=intent.ticker,
                        quantity=intent.quantity,
                        side=intent.side,
                        status="error",
                        client_order_id=intent.client_order_id,
                    )
                )

        return orders

    # -- get_positions ------------------------------------------------------

    def get_positions(self) -> List[BrokerPosition]:
        """Retrieve current positions, filtered by account."""
        self._ensure_connected()
        positions = self._client.positions()
        result: List[BrokerPosition] = []
        for pos in positions:
            # Filter by account if configured
            if self.config.account_id and pos.account != self.config.account_id:
                continue
            result.append(
                BrokerPosition(
                    ticker=pos.contract.localSymbol or pos.contract.symbol,
                    quantity=float(pos.position),
                    avg_cost=float(pos.avgCost),
                )
            )
        return result

    # -- get_orders ---------------------------------------------------------

    def get_orders(self, status: Optional[str] = None) -> List[Order]:
        """Retrieve open/recent orders."""
        self._ensure_connected()
        trades = self._client.trades()
        result: List[Order] = []
        for trade in trades:
            # Filter by account
            if self.config.account_id and trade.order.account != self.config.account_id:
                continue
            trade_status = trade.orderStatus.status if trade.orderStatus else "unknown"
            if status and trade_status.lower() != status.lower():
                continue
            result.append(
                Order(
                    asof=date.today(),
                    ticker=trade.contract.localSymbol or trade.contract.symbol,
                    quantity=float(trade.order.totalQuantity),
                    side=trade.order.action.lower(),
                    status=trade_status,
                    fill_price=float(trade.orderStatus.avgFillPrice) if trade.orderStatus else None,
                    broker_order_id=str(trade.order.orderId),
                )
            )
        return result

    # -- cancel_order -------------------------------------------------------

    def cancel_order(self, order_id: str) -> None:
        """Cancel an order by broker order id."""
        self._ensure_connected()
        int_id = int(order_id)
        if int_id in self._trade_map:
            trade = self._trade_map[int_id]
            self._client.cancel_order(trade.order)
            self._client.sleep(0.5)
            logger.info("Cancel requested for order %s", order_id)
        else:
            # Try to find in current trades
            for trade in self._client.trades():
                if trade.order.orderId == int_id:
                    self._client.cancel_order(trade.order)
                    self._client.sleep(0.5)
                    logger.info("Cancel requested for order %s (found in trades)", order_id)
                    return
            raise ValueError(f"Order {order_id} not found for cancellation")

    # -- get_account --------------------------------------------------------

    def get_account(self) -> BrokerAccount:
        """Retrieve real account summary (NLV, cash, buying power)."""
        self._ensure_connected()
        account = self.config.account_id or ""
        values = self._client.account_values(account=account)

        nlv = 0.0
        cash = 0.0
        buying_power = 0.0

        for v in values:
            if self.config.account_id and v.account != self.config.account_id:
                continue
            if v.tag == "NetLiquidation" and v.currency == "USD":
                nlv = float(v.value)
            elif v.tag == "AvailableFunds" and v.currency == "USD":
                cash = float(v.value)
            elif v.tag == "BuyingPower" and v.currency == "USD":
                buying_power = float(v.value)

        return BrokerAccount(
            asof=date.today(),
            cash=cash,
            equity=nlv,
            buying_power=buying_power,
        )

    # -- get_fills (new — not in BrokerBase ABC) ----------------------------

    def get_fills(
        self,
        time_min: Optional[datetime] = None,
        time_max: Optional[datetime] = None,
    ) -> List[Fill]:
        """Retrieve fills/executions for reconciliation.

        Parameters
        ----------
        time_min, time_max
            Optional UTC time bounds for filtering.

        Returns
        -------
        List[Fill]
            Internal Fill objects with price, qty, side, commission.
        """
        self._ensure_connected()
        ib_fills = self._client.fills()
        result: List[Fill] = []

        for f in ib_fills:
            # Filter by account
            if self.config.account_id and f.execution.acctNumber != self.config.account_id:
                continue

            # Parse execution time
            exec_time_str = f.execution.time
            try:
                exec_time = datetime.fromisoformat(str(exec_time_str))
            except (ValueError, TypeError):
                exec_time = None

            # Time filter
            if exec_time:
                if time_min and exec_time < time_min:
                    continue
                if time_max and exec_time > time_max:
                    continue

            # Commission
            commission = float(f.commissionReport.commission) if f.commissionReport else 0.0

            ticker = f.contract.localSymbol or f.contract.symbol
            side = f.execution.side.lower()  # "BOT" → need to normalize
            if side in ("bot", "bought"):
                side = "buy"
            elif side in ("sld", "sold"):
                side = "sell"

            result.append(
                Fill(
                    asof=date.today(),
                    ticker=ticker,
                    quantity=float(f.execution.shares),
                    price=float(f.execution.price),
                    side=side,
                    order_id=str(f.execution.orderId),
                    commission=commission,
                    execution_time=exec_time,
                )
            )

        return result
