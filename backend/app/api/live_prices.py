"""Live price quotes via Alpaca Market Data API.

Provides real-time last-trade prices for equities and futures,
with a 15-second in-memory cache to avoid rate limits.
All price fetches have strict timeouts to prevent blocking.
"""
from __future__ import annotations

import logging
import os
import time
import concurrent.futures
from typing import Any

import requests
from fastapi import APIRouter, Query

logger = logging.getLogger("algae.api.prices")

router = APIRouter(prefix="/api/prices", tags=["prices"])

# ── Alpaca config ────────────────────────────────────────────────────────
_ALPACA_KEY = os.environ.get("ALPACA_API_KEY", "")
_ALPACA_SECRET = os.environ.get("ALPACA_SECRET_KEY", "")
_ALPACA_DATA_URL = "https://data.alpaca.markets"

# ── In-memory price cache (symbol → (price, timestamp)) ─────────────────
_price_cache: dict[str, tuple[float, float]] = {}
_CACHE_TTL = 15.0  # seconds
_FETCH_TIMEOUT = 3  # max seconds per price fetch attempt

# ── Closing price cache (much longer TTL — closes don't change) ──────────
_close_cache: dict[str, tuple[float, float]] = {}
_CLOSE_CACHE_TTL = 4 * 3600  # 4 hours


def _alpaca_headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": _ALPACA_KEY,
        "APCA-API-SECRET-KEY": _ALPACA_SECRET,
        "Accept": "application/json",
    }


def _fetch_equity_quotes(symbols: list[str]) -> dict[str, float]:
    """Fetch latest trades for equity symbols from Alpaca v2."""
    if not symbols or not _ALPACA_KEY:
        return {}
    try:
        resp = requests.get(
            f"{_ALPACA_DATA_URL}/v2/stocks/trades/latest",
            headers=_alpaca_headers(),
            params={"symbols": ",".join(symbols), "feed": "iex"},
            timeout=_FETCH_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json().get("trades", {})
        return {sym: float(trade["p"]) for sym, trade in data.items() if "p" in trade}
    except Exception as exc:
        logger.warning("Alpaca equity quote failed: %s", exc)
        return {}


def _parse_futures_root(symbol: str) -> str | None:
    """Extract root from a futures symbol like RTYM6 → RTY."""
    import re
    m = re.match(r"^([A-Z0-9]{1,4}?)([FGHJKMNQUVXZ]\d{1,2})$", symbol)
    return m.group(1) if m else None


def _classify_symbols(symbols: list[str]) -> tuple[list[str], list[str]]:
    """Split symbols into (equities, futures)."""
    try:
        from sleeves.cooc_reversal_futures.contract_master import CONTRACT_MASTER
    except ImportError:
        CONTRACT_MASTER = {}

    equities, futures = [], []
    for sym in symbols:
        root = _parse_futures_root(sym)
        if root and root in CONTRACT_MASTER:
            futures.append(sym)
        else:
            equities.append(sym)
    return equities, futures


def get_live_prices(symbols: list[str]) -> dict[str, float]:
    """Return live prices for a list of symbols, using cache.
    
    All fetches are wrapped with strict timeouts to prevent blocking.
    """
    now = time.monotonic()
    result: dict[str, float] = {}
    stale: list[str] = []

    for sym in symbols:
        cached = _price_cache.get(sym)
        if cached and (now - cached[1]) < _CACHE_TTL:
            result[sym] = cached[0]
        else:
            stale.append(sym)

    if not stale:
        return result

    equities, futures = _classify_symbols(stale)

    # Fetch with a global timeout using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {}
        if equities:
            tasks["eq"] = pool.submit(_fetch_equity_quotes, equities)
        if futures:
            tasks["fut"] = pool.submit(_fetch_futures_quotes_yf, futures)

        for key, future in tasks.items():
            try:
                prices = future.result(timeout=_FETCH_TIMEOUT)
                for sym, price in prices.items():
                    _price_cache[sym] = (price, now)
                    result[sym] = price
            except concurrent.futures.TimeoutError:
                logger.warning("Price fetch '%s' timed out after %ds", key, _FETCH_TIMEOUT)
            except Exception as exc:
                logger.warning("Price fetch '%s' failed: %s", key, exc)

    return result


def get_closing_prices(symbols: list[str]) -> dict[str, float]:
    """Return previous-close prices for symbols.

    Uses a long-lived cache (4h) since closing prices don't change.
    Falls back gracefully — returns empty dict on failure.
    """
    now = time.monotonic()
    result: dict[str, float] = {}
    stale: list[str] = []

    for sym in symbols:
        cached = _close_cache.get(sym)
        if cached and (now - cached[1]) < _CLOSE_CACHE_TTL:
            result[sym] = cached[0]
        else:
            stale.append(sym)

    if not stale:
        return result

    equities, futures = _classify_symbols(stale)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        tasks = {}
        if equities:
            tasks["eq"] = pool.submit(_fetch_equity_closing, equities)
        if futures:
            tasks["fut"] = pool.submit(_fetch_futures_closing_yf, futures)

        for key, future in tasks.items():
            try:
                prices = future.result(timeout=_FETCH_TIMEOUT)
                for sym, price in prices.items():
                    _close_cache[sym] = (price, now)
                    result[sym] = price
            except concurrent.futures.TimeoutError:
                logger.warning("Close price fetch '%s' timed out", key)
            except Exception as exc:
                logger.warning("Close price fetch '%s' failed: %s", key, exc)

    return result


def _fetch_equity_closing(symbols: list[str]) -> dict[str, float]:
    """Fetch previous-close for equities via Alpaca snapshots."""
    if not symbols or not _ALPACA_KEY:
        # Fallback to yfinance for equities if no Alpaca key
        return _fetch_equity_closing_yf(symbols)
    try:
        resp = requests.get(
            f"{_ALPACA_DATA_URL}/v2/stocks/snapshots",
            headers=_alpaca_headers(),
            params={"symbols": ",".join(symbols), "feed": "iex"},
            timeout=_FETCH_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        prices: dict[str, float] = {}
        for sym, snap in data.items():
            daily = snap.get("dailyBar") or snap.get("prevDailyBar") or {}
            close = daily.get("c")
            if close and float(close) > 0:
                prices[sym] = float(close)
        return prices
    except Exception as exc:
        logger.warning("Alpaca snapshot failed: %s", exc)
        return _fetch_equity_closing_yf(symbols)


def _fetch_equity_closing_yf(symbols: list[str]) -> dict[str, float]:
    """Fallback: fetch previous close for equities via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        return {}
    prices: dict[str, float] = {}
    for sym in symbols:
        try:
            t = yf.Ticker(sym)
            close = getattr(t.fast_info, "previous_close", None)
            if close and close > 0:
                prices[sym] = float(close)
        except Exception:
            pass
    return prices


def _fetch_futures_closing_yf(symbols: list[str]) -> dict[str, float]:
    """Fetch previous close for futures via yfinance."""
    try:
        import yfinance as yf
    except ImportError:
        return {}
    prices: dict[str, float] = {}
    for sym in symbols:
        root = _parse_futures_root(sym)
        if not root:
            continue
        yf_ticker = f"{root}=F"
        try:
            t = yf.Ticker(yf_ticker)
            close = getattr(t.fast_info, "previous_close", None)
            if close and close > 0:
                prices[sym] = float(close)
        except Exception:
            pass
    return prices


def _fetch_futures_quotes_yf(symbols: list[str]) -> dict[str, float]:
    """Fallback: fetch futures prices via yfinance (free, no key).
    
    Has internal timeout protection — returns empty dict if yfinance hangs.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — futures prices unavailable")
        return {}

    prices: dict[str, float] = {}
    for sym in symbols:
        root = _parse_futures_root(sym)
        if not root:
            continue
        yf_ticker = f"{root}=F"
        try:
            ticker = yf.Ticker(yf_ticker)
            price = getattr(ticker.fast_info, "last_price", None) or getattr(ticker.fast_info, "previous_close", None)
            if price and price > 0:
                prices[sym] = float(price)
        except Exception as exc:
            logger.debug("yfinance %s failed: %s", yf_ticker, exc)
    return prices


@router.get("/quotes")
def get_quotes(
    symbols: str = Query(..., description="Comma-separated symbols"),
) -> dict[str, Any]:
    """Return latest prices for requested symbols."""
    sym_list = [s.strip().upper() for s in symbols.split(",") if s.strip()]
    prices = get_live_prices(sym_list)
    return {"prices": prices, "count": len(prices)}
