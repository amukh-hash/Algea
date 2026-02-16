"""Thin wrapper around ib_insync to isolate IBKR library details.

All ib_insync imports are confined to this module.  If the library is
replaced later (e.g. with ``ib_async``), only this file changes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd

from ib_insync import (  # type: ignore[import-untyped]
    IB,
    Contract,
    Future,
    Order as IbOrder,
    Trade,
    Fill as IbFill,
    Execution,
    AccountValue,
    Position,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thin wrapper
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualifiedContract:
    """Snapshot of a successfully qualified IBKR contract."""

    con_id: int
    symbol: str
    local_symbol: str
    exchange: str
    currency: str
    multiplier: str
    expiry: str  # YYYYMMDD or YYYYMM
    trading_class: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "con_id": self.con_id,
            "symbol": self.symbol,
            "local_symbol": self.local_symbol,
            "exchange": self.exchange,
            "currency": self.currency,
            "multiplier": self.multiplier,
            "expiry": self.expiry,
            "trading_class": self.trading_class,
        }


class IbkrClient:
    """Thin async-aware wrapper around :class:`ib_insync.IB`.

    Parameters
    ----------
    host : IBKR gateway/TWS host
    port : gateway/TWS port
    client_id : IBKR client id (must be unique across connections)
    timeout : seconds to wait for connection
    readonly : if True, order placement is blocked at this layer
    max_retries : max connection attempts
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 17,
        timeout: int = 20,
        readonly: bool = False,
        max_retries: int = 3,
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.readonly = readonly
        self.max_retries = max_retries
        self._ib = IB()

    # -- connection lifecycle -----------------------------------------------

    @property
    def is_connected(self) -> bool:
        return self._ib.isConnected()

    def connect(self) -> None:
        """Connect with bounded retry."""
        for attempt in range(1, self.max_retries + 1):
            try:
                logger.info(
                    "IBKR connect attempt %d/%d → %s:%d (clientId=%d, readonly=%s)",
                    attempt,
                    self.max_retries,
                    self.host,
                    self.port,
                    self.client_id,
                    self.readonly,
                )
                self._ib.connect(
                    self.host,
                    self.port,
                    clientId=self.client_id,
                    timeout=self.timeout,
                    readonly=self.readonly,
                )
                logger.info("IBKR connected (serverVersion=%s)", self._ib.client.serverVersion())
                return
            except Exception as exc:
                logger.warning("IBKR connect attempt %d failed: %s", attempt, exc)
                if attempt == self.max_retries:
                    raise ConnectionError(
                        f"Failed to connect to IBKR at {self.host}:{self.port} "
                        f"after {self.max_retries} attempts"
                    ) from exc

    def disconnect(self) -> None:
        if self._ib.isConnected():
            self._ib.disconnect()
            logger.info("IBKR disconnected")

    def ensure_connected(self) -> None:
        if not self._ib.isConnected():
            self.connect()

    # -- contract qualification ---------------------------------------------

    def qualify_contracts(self, *contracts: Contract) -> List[Contract]:
        """Qualify one or more contracts.  Returns qualified copies."""
        self.ensure_connected()
        return self._ib.qualifyContracts(*contracts)

    # -- orders --------------------------------------------------------------

    def place_order(self, contract: Contract, order: IbOrder) -> Trade:
        """Place a single order.  Raises if readonly."""
        if self.readonly:
            raise PermissionError("IbkrClient is in readonly mode — order placement blocked")
        self.ensure_connected()
        return self._ib.placeOrder(contract, order)

    def cancel_order(self, order: IbOrder) -> Trade:
        self.ensure_connected()
        return self._ib.cancelOrder(order)

    # -- queries -------------------------------------------------------------

    def positions(self) -> List[Position]:
        self.ensure_connected()
        return self._ib.positions()

    def open_orders(self) -> List[Trade]:
        self.ensure_connected()
        return self._ib.openTrades()

    def trades(self) -> List[Trade]:
        self.ensure_connected()
        return self._ib.trades()

    def fills(self) -> List[IbFill]:
        self.ensure_connected()
        return self._ib.fills()

    def executions(self) -> List[Execution]:
        self.ensure_connected()
        return [f.execution for f in self._ib.fills()]

    def account_summary(self, account: str = "") -> List[AccountValue]:
        self.ensure_connected()
        return self._ib.accountSummary(account=account)

    def account_values(self, account: str = "") -> List[AccountValue]:
        self.ensure_connected()
        return self._ib.accountValues(account=account)

    def server_time(self) -> datetime:
        self.ensure_connected()
        return self._ib.reqCurrentTime()

    # -- historical data -----------------------------------------------------

    def historical_bars(
        self,
        contract: Contract,
        duration: str = "60 D",
        bar_size: str = "1 day",
        what_to_show: str = "TRADES",
        use_rth: bool = True,
        end_dt: str = "",
    ) -> pd.DataFrame:
        """Fetch historical bars via ``reqHistoricalData``.

        Parameters
        ----------
        contract
            Qualified IBKR contract.
        duration
            How far back to look (e.g. ``"60 D"``, ``"1 Y"``).
        bar_size
            Bar granularity (e.g. ``"1 day"``, ``"1 hour"``).
        what_to_show
            ``"TRADES"``, ``"MIDPOINT"``, ``"BID"``, ``"ASK"``.
        use_rth
            If True, only regular-trading-hours bars.
        end_dt
            End datetime (empty = now).

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp, open, high, low, close, volume``.
        """
        self.ensure_connected()
        bars = self._ib.reqHistoricalData(
            contract,
            endDateTime=end_dt,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow=what_to_show,
            useRTH=use_rth,
            formatDate=1,
        )
        if not bars:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        records = []
        for bar in bars:
            records.append({
                "timestamp": pd.Timestamp(bar.date, tz="UTC"),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": int(bar.volume),
            })
        return pd.DataFrame(records)

    # -- sleep / event loop --------------------------------------------------

    def sleep(self, secs: float = 0.5) -> None:
        """Let ib_insync event loop process pending events."""
        self._ib.sleep(secs)

