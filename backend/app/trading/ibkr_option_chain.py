"""IBKR option chain fetcher for the VRP sleeve.

Produces a normalized DataFrame in the schema that ``VRPStrategy`` expects,
with rate-limit-safe batching and strict filtering guardrails.

Required output columns (consumed by IVSurfaceBuilder + VRPStrategy):
    strike, expiry, dte, option_type, implied_vol, mid, bid, ask,
    risk_free_rate, dividend_yield, underlying_price, open_interest, volume
"""
from __future__ import annotations

import logging
import math
import time
from datetime import date, datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Dividend yield estimates (annual) — update periodically
_DIVIDEND_YIELDS: Dict[str, float] = {
    "SPY": 0.013,
    "QQQ": 0.006,
    "IWM": 0.012,
}

# Default risk-free rate (annualized)
_DEFAULT_RISK_FREE_RATE: float = 0.045

# Pacing constants
_BATCH_SIZE: int = 40  # contracts per snapshot batch
_BATCH_SLEEP: float = 0.5  # seconds between batches
_SNAPSHOT_WAIT: float = 2.0  # seconds to wait per snapshot batch


def fetch_option_chain(
    ib,  # ib_insync.IB
    underlying: str,
    *,
    asof_date: Optional[date] = None,
    dte_range: Tuple[int, int] = (20, 60),
    strike_band_pct: float = 0.15,
    max_expiries: int = 2,
    max_strikes_per_expiry: int = 30,
    snapshot: bool = True,
    risk_free_rate: Optional[float] = None,
) -> pd.DataFrame:
    """Fetch option chain from IBKR, returning a normalized DataFrame.

    Parameters
    ----------
    ib : ib_insync.IB
        Connected IB instance.
    underlying : str
        Ticker symbol (e.g. "SPY").
    asof_date : date, optional
        Reference date for DTE computation (default: today).
    dte_range : tuple
        Min/max days to expiry for expiry filtering.
    strike_band_pct : float
        Keep strikes within ``spot * (1 ± strike_band_pct)``.
    max_expiries : int
        Max number of expiries to include (nearest first).
    max_strikes_per_expiry : int
        Max strikes per expiry per option type (puts/calls each capped).
    snapshot : bool
        If True, use snapshot market data (frozen data, no streaming).
    risk_free_rate : float, optional
        Override risk-free rate.

    Returns
    -------
    pd.DataFrame
        Columns: strike, expiry, dte, option_type, implied_vol, mid, bid,
        ask, risk_free_rate, dividend_yield, underlying_price, open_interest,
        volume.  Empty rows (missing IV or bid/ask) are dropped.
    """
    from ib_insync import Stock, Option  # type: ignore[import-untyped]

    if not ib.isConnected():
        raise ConnectionError("IB Gateway is not connected")

    ref_date = asof_date or date.today()
    rfr = risk_free_rate if risk_free_rate is not None else _DEFAULT_RISK_FREE_RATE
    div_yield = _DIVIDEND_YIELDS.get(underlying, 0.01)

    # ------------------------------------------------------------------
    # 1. Get current spot price
    # ------------------------------------------------------------------
    stock = Stock(underlying, "SMART", "USD")
    qualified_stocks = ib.qualifyContracts(stock)
    if not qualified_stocks or qualified_stocks[0].conId == 0:
        raise ValueError(f"Cannot qualify stock contract for {underlying}")

    stock_contract = qualified_stocks[0]
    spot_ticker = ib.reqMktData(stock_contract, "", True, False)
    ib.sleep(1.5)

    spot = _extract_price(spot_ticker)
    if spot is None or spot <= 0:
        raise ValueError(f"No spot price for {underlying}: {spot_ticker}")

    logger.info("fetch_option_chain(%s): spot=%.2f", underlying, spot)

    # ------------------------------------------------------------------
    # 2. Get available expiries and strikes via reqSecDefOptParams
    # ------------------------------------------------------------------
    chains = ib.reqSecDefOptParams(
        underlyingSymbol=underlying,
        futFopExchange="",
        underlyingSecType="STK",
        underlyingConId=stock_contract.conId,
    )

    if not chains:
        logger.warning("No option chain definitions returned for %s", underlying)
        return _empty_chain_df()

    # Use SMART exchange chain (or first available)
    chain_def = None
    for c in chains:
        if c.exchange == "SMART":
            chain_def = c
            break
    if chain_def is None:
        chain_def = chains[0]

    # ------------------------------------------------------------------
    # 3. Filter expiries to dte_range, keep nearest N
    # ------------------------------------------------------------------
    dte_min, dte_max = dte_range
    valid_expiries: List[Tuple[date, int]] = []
    for exp_str in chain_def.expirations:
        exp_date = _parse_expiry(exp_str)
        if exp_date is None:
            continue
        dte = (exp_date - ref_date).days
        if dte_min <= dte <= dte_max:
            valid_expiries.append((exp_date, dte))

    valid_expiries.sort(key=lambda x: x[1])  # nearest first
    selected_expiries = valid_expiries[:max_expiries]

    if not selected_expiries:
        logger.warning(
            "No expiries in DTE range [%d, %d] for %s (available: %d)",
            dte_min, dte_max, underlying, len(chain_def.expirations),
        )
        return _empty_chain_df()

    logger.info(
        "Selected %d expiries for %s: %s",
        len(selected_expiries), underlying,
        [(e[0].isoformat(), e[1]) for e in selected_expiries],
    )

    # ------------------------------------------------------------------
    # 4. Filter strikes to band around spot, cap count
    # ------------------------------------------------------------------
    strike_lo = spot * (1 - strike_band_pct)
    strike_hi = spot * (1 + strike_band_pct)
    all_strikes = sorted(
        s for s in chain_def.strikes if strike_lo <= s <= strike_hi
    )

    # Keep nearest N strikes centered on spot
    if len(all_strikes) > max_strikes_per_expiry:
        # Find center index closest to spot
        center_idx = min(range(len(all_strikes)), key=lambda i: abs(all_strikes[i] - spot))
        half = max_strikes_per_expiry // 2
        start = max(0, center_idx - half)
        end = min(len(all_strikes), start + max_strikes_per_expiry)
        start = max(0, end - max_strikes_per_expiry)
        all_strikes = all_strikes[start:end]

    logger.info(
        "Strikes for %s: %d in [%.1f, %.1f] (band ±%.0f%%)",
        underlying, len(all_strikes), strike_lo, strike_hi,
        strike_band_pct * 100,
    )

    # ------------------------------------------------------------------
    # 5. Build Option contracts (puts + calls)
    # ------------------------------------------------------------------
    option_contracts: List[Any] = []
    contract_meta: List[Dict[str, Any]] = []

    for exp_date, dte in selected_expiries:
        exp_str = exp_date.strftime("%Y%m%d")
        for strike in all_strikes:
            for right in ("P", "C"):
                opt = Option(underlying, exp_str, strike, right, "SMART")
                option_contracts.append(opt)
                contract_meta.append({
                    "strike": strike,
                    "expiry": exp_date,
                    "dte": dte,
                    "option_type": "put" if right == "P" else "call",
                    "right": right,
                })

    total_requested = len(option_contracts)
    logger.info(
        "Built %d option contracts for %s (%d expiries × %d strikes × 2 types)",
        total_requested, underlying, len(selected_expiries), len(all_strikes),
    )

    # ------------------------------------------------------------------
    # 6. Qualify contracts in batches
    # ------------------------------------------------------------------
    qualified_map: Dict[int, Any] = {}  # index → qualified contract
    for batch_start in range(0, len(option_contracts), _BATCH_SIZE):
        batch = option_contracts[batch_start:batch_start + _BATCH_SIZE]
        try:
            result = ib.qualifyContracts(*batch)
            for i, qc in enumerate(result):
                if qc.conId and qc.conId != 0:
                    qualified_map[batch_start + i] = qc
        except Exception as exc:
            logger.warning(
                "Qualify batch %d–%d failed: %s",
                batch_start, batch_start + len(batch) - 1, exc,
            )
        if batch_start + _BATCH_SIZE < len(option_contracts):
            time.sleep(0.1)

    logger.info(
        "Qualified %d / %d option contracts for %s",
        len(qualified_map), total_requested, underlying,
    )

    if not qualified_map:
        return _empty_chain_df()

    # ------------------------------------------------------------------
    # 7. Request snapshot market data in throttled batches
    # ------------------------------------------------------------------
    tickers_by_idx: Dict[int, Any] = {}
    qualified_indices = sorted(qualified_map.keys())

    for batch_start in range(0, len(qualified_indices), _BATCH_SIZE):
        batch_indices = qualified_indices[batch_start:batch_start + _BATCH_SIZE]
        batch_tickers = []

        for idx in batch_indices:
            qc = qualified_map[idx]
            # snapshot=True for frozen data, regulatorySnapshot=False
            tk = ib.reqMktData(qc, "", snapshot, False)
            batch_tickers.append((idx, tk))

        # Wait for snapshots to arrive
        ib.sleep(_SNAPSHOT_WAIT)

        for idx, tk in batch_tickers:
            tickers_by_idx[idx] = tk

        # Throttle between batches
        if batch_start + _BATCH_SIZE < len(qualified_indices):
            time.sleep(_BATCH_SLEEP)

    # ------------------------------------------------------------------
    # 8. Extract quotes and build DataFrame
    # ------------------------------------------------------------------
    rows: List[Dict[str, Any]] = []
    dropped_no_iv = 0
    dropped_no_quote = 0

    for idx, ticker in tickers_by_idx.items():
        meta = contract_meta[idx]

        bid = _safe_float(ticker.bid)
        ask = _safe_float(ticker.ask)

        # Skip if no usable bid/ask
        if bid is None and ask is None:
            dropped_no_quote += 1
            continue

        # Use available side(s)
        if bid is not None and ask is not None:
            mid = (bid + ask) / 2.0
        elif bid is not None:
            mid = bid
        else:
            mid = ask

        # Default missing side to mid
        bid = bid if bid is not None else mid
        ask = ask if ask is not None else mid

        # Extract implied vol from model Greeks
        iv = None
        if ticker.modelGreeks and hasattr(ticker.modelGreeks, "impliedVol"):
            iv_raw = ticker.modelGreeks.impliedVol
            if iv_raw is not None and not math.isnan(iv_raw) and iv_raw > 0:
                iv = float(iv_raw)

        if iv is None:
            dropped_no_iv += 1
            continue

        # Extract OI and volume (best effort)
        oi = int(ticker.openInterest) if _is_valid(ticker.openInterest) else 0
        vol = int(ticker.volume) if _is_valid(ticker.volume) else 0

        rows.append({
            "strike": meta["strike"],
            "expiry": meta["expiry"],
            "dte": meta["dte"],
            "option_type": meta["option_type"],
            "implied_vol": iv,
            "mid": mid,
            "bid": bid,
            "ask": ask,
            "risk_free_rate": rfr,
            "dividend_yield": div_yield,
            "underlying_price": spot,
            "open_interest": oi,
            "volume": vol,
        })

    logger.info(
        "Chain for %s: %d usable rows, %d dropped (no IV), %d dropped (no quote)",
        underlying, len(rows), dropped_no_iv, dropped_no_quote,
    )

    if not rows:
        return _empty_chain_df()

    df = pd.DataFrame(rows)
    df["expiry"] = pd.to_datetime(df["expiry"])
    df = df.sort_values(["expiry", "option_type", "strike"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_price(ticker) -> Optional[float]:  # type: ignore[no-untyped-def]
    """Extract a usable price from a snapshot ticker."""
    if ticker.last and not math.isnan(ticker.last) and ticker.last > 0:
        return float(ticker.last)
    if ticker.close and not math.isnan(ticker.close) and ticker.close > 0:
        return float(ticker.close)
    if (
        ticker.bid and ticker.ask
        and not math.isnan(ticker.bid)
        and not math.isnan(ticker.ask)
        and ticker.bid > 0 and ticker.ask > 0
    ):
        return (float(ticker.bid) + float(ticker.ask)) / 2.0
    return None


def _safe_float(val) -> Optional[float]:  # type: ignore[no-untyped-def]
    """Return float if val is a valid positive number, else None."""
    if val is None:
        return None
    try:
        f = float(val)
        if math.isnan(f) or f <= 0:
            return None
        return f
    except (TypeError, ValueError):
        return None


def _is_valid(val) -> bool:  # type: ignore[no-untyped-def]
    """Return True if val is a valid non-negative number."""
    if val is None:
        return False
    try:
        f = float(val)
        return not math.isnan(f) and f >= 0
    except (TypeError, ValueError):
        return False


def _parse_expiry(exp_str: str) -> Optional[date]:
    """Parse IBKR expiry string (YYYYMMDD) to date."""
    try:
        return datetime.strptime(exp_str, "%Y%m%d").date()
    except (ValueError, TypeError):
        return None


def _empty_chain_df() -> pd.DataFrame:
    """Return an empty DataFrame with the expected chain schema."""
    return pd.DataFrame(columns=[
        "strike", "expiry", "dte", "option_type", "implied_vol",
        "mid", "bid", "ask", "risk_free_rate", "dividend_yield",
        "underlying_price", "open_interest", "volume",
    ])
