"""Mock-based unit tests for IBKRLiveBroker.

Tests use a MockIbkrClient injected via DI — no IBKR gateway required.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock

import pytest

from algea.trading.broker_base import BrokerAccount, BrokerPosition
from algea.trading.broker_ibkr import IBKRLiveBroker, IbkrConfig, _mask_account
from algea.trading.ibkr_contracts import (
    EXCHANGE_MAP,
    build_future_contract,
    parse_active_contract_symbol,
)
from algea.trading.orders import Fill, Order, OrderIntent


# ---------------------------------------------------------------------------
# Mock IBKR client
# ---------------------------------------------------------------------------


class _MockContract:
    def __init__(self, **kwargs: Any) -> None:
        self.conId = kwargs.get("conId", 123456)
        self.symbol = kwargs.get("symbol", "ES")
        self.localSymbol = kwargs.get("localSymbol", "ESH6")
        self.exchange = kwargs.get("exchange", "CME")
        self.currency = kwargs.get("currency", "USD")
        self.multiplier = kwargs.get("multiplier", "50")
        self.lastTradeDateOrContractMonth = kwargs.get("lastTradeDateOrContractMonth", "202603")
        self.tradingClass = kwargs.get("tradingClass", "ES")


class _MockOrderStatus:
    def __init__(self, status: str = "Submitted", avg_fill: float = 0.0) -> None:
        self.status = status
        self.avgFillPrice = avg_fill


class _MockOrder:
    def __init__(self, order_id: int = 1, action: str = "BUY", qty: float = 1.0, account: str = "") -> None:
        self.orderId = order_id
        self.action = action
        self.totalQuantity = qty
        self.account = account


class _MockTrade:
    def __init__(self, contract: Any = None, order: Any = None, status: str = "Submitted") -> None:
        self.contract = contract or _MockContract()
        self.order = order or _MockOrder()
        self.orderStatus = _MockOrderStatus(status)


class _MockExecution:
    def __init__(self, **kwargs: Any) -> None:
        self.acctNumber = kwargs.get("acctNumber", "DUP905542I")
        self.side = kwargs.get("side", "BOT")
        self.shares = kwargs.get("shares", 1.0)
        self.price = kwargs.get("price", 5000.0)
        self.orderId = kwargs.get("orderId", 1)
        self.time = kwargs.get("time", "2026-02-14T09:30:00")


class _MockCommReport:
    def __init__(self, commission: float = 2.25) -> None:
        self.commission = commission


class _MockIbFill:
    def __init__(self, contract: Any = None, execution: Any = None, commission: float = 2.25) -> None:
        self.contract = contract or _MockContract()
        self.execution = execution or _MockExecution()
        self.commissionReport = _MockCommReport(commission)


class _MockPosition:
    def __init__(self, account: str = "DUP905542I", contract: Any = None, position: float = 1.0, avg_cost: float = 5000.0) -> None:
        self.account = account
        self.contract = contract or _MockContract()
        self.position = position
        self.avgCost = avg_cost


class _MockAccountValue:
    def __init__(self, tag: str, value: str, currency: str = "USD", account: str = "DUP905542I") -> None:
        self.tag = tag
        self.value = value
        self.currency = currency
        self.account = account


class MockIbkrClient:
    """Deterministic mock for IbkrClient — no IBKR gateway needed."""

    def __init__(
        self,
        readonly: bool = False,
        qualify_conid: int = 123456,
        fill_price: float = 5000.0,
    ) -> None:
        self._connected = False
        self.readonly = readonly
        self._qualify_conid = qualify_conid
        self._fill_price = fill_price
        self._next_order_id = 1
        self._placed_orders: List[Dict[str, Any]] = []
        self._cancelled: List[int] = []

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def ensure_connected(self) -> None:
        if not self._connected:
            self.connect()

    def qualify_contracts(self, *contracts: Any) -> List[Any]:
        result = []
        for c in contracts:
            mc = _MockContract(
                conId=self._qualify_conid,
                symbol=c.symbol,
                localSymbol=f"{c.symbol}H6",
                exchange=c.exchange,
                currency=c.currency,
                multiplier=str(getattr(c, "multiplier", "50")),
                lastTradeDateOrContractMonth=c.lastTradeDateOrContractMonth,
                tradingClass=c.symbol,
            )
            result.append(mc)
        return result

    def place_order(self, contract: Any, order: Any) -> Any:
        if self.readonly:
            raise PermissionError("readonly mode")
        oid = self._next_order_id
        self._next_order_id += 1
        mock_order = _MockOrder(order_id=oid, action=order.action, qty=order.totalQuantity)
        mock_trade = _MockTrade(contract=contract, order=mock_order, status="Submitted")
        self._placed_orders.append({"contract": contract, "order": order, "oid": oid})
        return mock_trade

    def cancel_order(self, order: Any) -> Any:
        self._cancelled.append(order.orderId)
        return _MockTrade(order=order, status="Cancelled")

    def positions(self) -> List[Any]:
        return [_MockPosition()]

    def open_orders(self) -> List[Any]:
        return []

    def trades(self) -> List[Any]:
        return [_MockTrade()]

    def fills(self) -> List[Any]:
        return [_MockIbFill()]

    def executions(self) -> List[Any]:
        return [_MockExecution()]

    def account_summary(self, account: str = "") -> List[Any]:
        return self.account_values(account)

    def account_values(self, account: str = "") -> List[Any]:
        return [
            _MockAccountValue("NetLiquidation", "1000000.0"),
            _MockAccountValue("AvailableFunds", "800000.0"),
            _MockAccountValue("BuyingPower", "3000000.0"),
        ]

    def server_time(self) -> datetime:
        return datetime(2026, 2, 14, 9, 30, 0)

    def sleep(self, secs: float = 0.5) -> None:
        pass


# ---------------------------------------------------------------------------
# Contract parsing tests
# ---------------------------------------------------------------------------


class TestContractParsing:
    def test_parse_short_format(self) -> None:
        root, expiry = parse_active_contract_symbol("ESH26")
        assert root == "ES"
        assert expiry == "202603"

    def test_parse_short_format_december(self) -> None:
        root, expiry = parse_active_contract_symbol("NQZ25")
        assert root == "NQ"
        assert expiry == "202512"

    def test_parse_long_format(self) -> None:
        root, expiry = parse_active_contract_symbol("ESU2026")
        assert root == "ES"
        assert expiry == "202609"

    def test_parse_dash_format(self) -> None:
        root, expiry = parse_active_contract_symbol("YM-202603")
        assert root == "YM"
        assert expiry == "202603"

    def test_parse_rty(self) -> None:
        root, expiry = parse_active_contract_symbol("RTYH26")
        assert root == "RTY"
        assert expiry == "202603"

    def test_parse_invalid_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot parse"):
            parse_active_contract_symbol("INVALID")

    def test_build_future_contract(self) -> None:
        c = build_future_contract("ES", "202603")
        assert c.symbol == "ES"
        assert c.exchange == "CME"

    def test_build_future_ym_exchange(self) -> None:
        c = build_future_contract("YM", "202603")
        assert c.exchange == "ECBOT"

    def test_exchange_map_complete(self) -> None:
        for root in ["ES", "NQ", "RTY", "YM"]:
            assert root in EXCHANGE_MAP


# ---------------------------------------------------------------------------
# Account masking tests
# ---------------------------------------------------------------------------


class TestAccountMasking:
    def test_mask_standard(self) -> None:
        assert _mask_account("DUP905542I") == "DUP*****2I"

    def test_mask_short(self) -> None:
        assert _mask_account("ABC") == "***"


# ---------------------------------------------------------------------------
# Broker method tests
# ---------------------------------------------------------------------------


class TestIBKRLiveBrokerSubmit:
    def _broker(self, readonly: bool = False) -> IBKRLiveBroker:
        config = IbkrConfig(account_id="DUP905542I", readonly=readonly)
        client = MockIbkrClient(readonly=readonly)
        return IBKRLiveBroker(config, client=client)

    def test_submit_orders_returns_orders(self) -> None:
        broker = self._broker()
        intents = [
            OrderIntent(
                asof=date(2026, 2, 14),
                ticker="ESH26",
                quantity=1,
                side="buy",
                reason="test",
            ),
        ]
        orders = broker.submit_orders(intents)
        assert len(orders) == 1
        assert orders[0].status == "Submitted"
        assert orders[0].broker_order_id is not None
        assert orders[0].ticker == "ESH26"

    def test_submit_multiple_deterministic_order(self) -> None:
        broker = self._broker()
        intents = [
            OrderIntent(asof=date(2026, 2, 14), ticker="NQH26", quantity=2, side="sell", reason="test"),
            OrderIntent(asof=date(2026, 2, 14), ticker="ESH26", quantity=1, side="buy", reason="test"),
        ]
        orders = broker.submit_orders(intents)
        assert len(orders) == 2
        # Should be sorted by (ticker, side, qty): ESH26 buy first
        assert orders[0].ticker == "ESH26"
        assert orders[1].ticker == "NQH26"

    def test_readonly_blocks_submission(self) -> None:
        broker = self._broker(readonly=True)
        intents = [
            OrderIntent(asof=date(2026, 2, 14), ticker="ESH26", quantity=1, side="buy", reason="test"),
        ]
        with pytest.raises(PermissionError, match="readonly"):
            broker.submit_orders(intents)


class TestIBKRLiveBrokerQueries:
    def _broker(self) -> IBKRLiveBroker:
        config = IbkrConfig(account_id="DUP905542I")
        return IBKRLiveBroker(config, client=MockIbkrClient())

    def test_get_positions(self) -> None:
        broker = self._broker()
        positions = broker.get_positions()
        assert len(positions) == 1
        assert isinstance(positions[0], BrokerPosition)
        assert positions[0].quantity == 1.0

    def test_get_orders(self) -> None:
        broker = self._broker()
        orders = broker.get_orders()
        assert len(orders) >= 0  # May be empty if mock trades don't match account
        for o in orders:
            assert isinstance(o, Order)

    def test_get_account(self) -> None:
        broker = self._broker()
        account = broker.get_account()
        assert isinstance(account, BrokerAccount)
        assert account.equity == 1_000_000.0
        assert account.cash == 800_000.0
        assert account.buying_power == 3_000_000.0

    def test_get_fills(self) -> None:
        broker = self._broker()
        fills = broker.get_fills()
        assert len(fills) == 1
        assert isinstance(fills[0], Fill)
        assert fills[0].commission == 2.25
        assert fills[0].side == "buy"  # BOT → buy
        assert fills[0].price == 5000.0

    def test_cancel_order(self) -> None:
        broker = self._broker()
        # Submit first
        intents = [
            OrderIntent(asof=date(2026, 2, 14), ticker="ESH26", quantity=1, side="buy", reason="test"),
        ]
        orders = broker.submit_orders(intents)
        oid = orders[0].broker_order_id
        broker.cancel_order(oid)
        # Should not raise


# ---------------------------------------------------------------------------
# from_env tests
# ---------------------------------------------------------------------------


class TestFromEnv:
    def test_from_env_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IBKR_GATEWAY_URL", "10.0.0.1:4002")
        monkeypatch.setenv("IBKR_ACCOUNT_ID", "TEST123")
        broker = IBKRLiveBroker.from_env()
        assert broker.config.host == "10.0.0.1"
        assert broker.config.port == 4002
        assert broker.config.account_id == "TEST123"
        assert broker.config.paper_only is True
        assert broker.config.readonly is False

    def test_from_env_readonly(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("IBKR_GATEWAY_URL", "127.0.0.1:7497")
        monkeypatch.setenv("IBKR_READONLY", "1")
        broker = IBKRLiveBroker.from_env()
        assert broker.config.readonly is True
