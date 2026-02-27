"""IBKR market data helpers for the VRP sleeve.

Provides VIX series, underlying close prices, and current spot quotes
required by ``VRPStrategy.compute_features()`` and ``predict()``.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def _ensure_connected(ib) -> None:  # type: ignore[no-untyped-def]
    """Ensure the ib_insync IB instance is connected."""
    if not ib.isConnected():
        raise ConnectionError("IB Gateway is not connected — cannot fetch market data")


def fetch_underlying_closes(
    ib,  # ib_insync.IB
    symbol: str,
    *,
    lookback_days: int = 252,
) -> pd.Series:
    """Fetch daily close prices for *symbol* over the trailing *lookback_days*.

    Returns a ``pd.Series`` indexed by date with monotonic ascending index.
    """
    from ib_insync import Stock  # type: ignore[import-untyped]

    _ensure_connected(ib)

    contract = Stock(symbol, "SMART", "USD")
    qualified = ib.qualifyContracts(contract)
    if not qualified or qualified[0].conId == 0:
        raise ValueError(f"Could not qualify stock contract for {symbol}")

    bars = ib.reqHistoricalData(
        qualified[0],
        endDateTime="",
        durationStr=f"{lookback_days} D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )

    if not bars:
        raise ValueError(f"No historical bars returned for {symbol}")

    records = []
    for bar in bars:
        records.append({
            "date": pd.Timestamp(bar.date).normalize(),
            "close": float(bar.close),
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date")
    series = df.set_index("date")["close"]
    series.name = symbol

    logger.info(
        "fetch_underlying_closes(%s): %d bars, range %s → %s",
        symbol, len(series),
        series.index[0].date() if len(series) else "N/A",
        series.index[-1].date() if len(series) else "N/A",
    )
    return series


def fetch_vix_series(
    ib,  # ib_insync.IB
    *,
    lookback_days: int = 252,
) -> pd.Series:
    """Fetch daily VIX closes over trailing *lookback_days*.

    Uses the CBOE VIX index contract.
    Returns a ``pd.Series`` indexed by date.
    """
    from ib_insync import Index  # type: ignore[import-untyped]

    _ensure_connected(ib)

    contract = Index("VIX", "CBOE", "USD")
    qualified = ib.qualifyContracts(contract)
    if not qualified or qualified[0].conId == 0:
        raise ValueError("Could not qualify VIX index contract")

    bars = ib.reqHistoricalData(
        qualified[0],
        endDateTime="",
        durationStr=f"{lookback_days} D",
        barSizeSetting="1 day",
        whatToShow="TRADES",
        useRTH=True,
        formatDate=1,
    )

    if not bars:
        raise ValueError("No historical bars returned for VIX")

    records = []
    for bar in bars:
        records.append({
            "date": pd.Timestamp(bar.date).normalize(),
            "close": float(bar.close),
        })

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset=["date"], keep="last")
    df = df.sort_values("date")
    series = df.set_index("date")["close"]
    series.name = "VIX"

    logger.info(
        "fetch_vix_series: %d bars, range %s → %s",
        len(series),
        series.index[0].date() if len(series) else "N/A",
        series.index[-1].date() if len(series) else "N/A",
    )
    return series


def get_current_price(ib, symbol: str) -> float:  # type: ignore[no-untyped-def]
    """Return the latest market price for *symbol*.

    Uses a snapshot quote.  Raises ``ValueError`` if no price is available.
    """
    from ib_insync import Stock  # type: ignore[import-untyped]

    _ensure_connected(ib)

    contract = Stock(symbol, "SMART", "USD")
    qualified = ib.qualifyContracts(contract)
    if not qualified or qualified[0].conId == 0:
        raise ValueError(f"Could not qualify stock contract for {symbol}")

    ticker = ib.reqMktData(qualified[0], "", True, False)
    ib.sleep(1.0)  # allow snapshot to arrive

    price: Optional[float] = None
    # Prefer last, then close, then mid of bid/ask
    if ticker.last and ticker.last > 0:
        price = float(ticker.last)
    elif ticker.close and ticker.close > 0:
        price = float(ticker.close)
    elif ticker.bid and ticker.ask and ticker.bid > 0 and ticker.ask > 0:
        price = (float(ticker.bid) + float(ticker.ask)) / 2.0

    if price is None or price <= 0:
        raise ValueError(f"No valid price available for {symbol} — ticker: {ticker}")

    logger.info("get_current_price(%s) = %.4f", symbol, price)
    return price
