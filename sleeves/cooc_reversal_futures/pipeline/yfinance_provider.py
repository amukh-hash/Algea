"""Yahoo Finance data provider for the pipeline's bronze ingestion layer.

Extends :class:`FuturesDataProvider` to fetch daily bars from Yahoo Finance
using continuous futures tickers (e.g. ``ES=F``, ``GC=F``).  This provides
free access to 10+ years of daily OHLCV data for major futures contracts.

Best for research/exploration runs.  For promotion-grade runs, use the
IBKR historical provider with a live account.
"""
from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .ingest import FuturesDataProvider

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Root → Yahoo Finance ticker mapping
# ---------------------------------------------------------------------------
# Yahoo uses "{ROOT}=F" for most CME/COMEX/NYMEX/CBOT continuous contracts.
# Some roots need special mapping.

YFINANCE_TICKER_MAP: Dict[str, str] = {
    # Equity indices
    "ES":  "ES=F",
    "NQ":  "NQ=F",
    "YM":  "YM=F",
    "RTY": "RTY=F",
    # Energy
    "CL":  "CL=F",
    # Metals
    "GC":  "GC=F",
    "SI":  "SI=F",
    "HG":  "HG=F",
    # Rates
    "ZN":  "ZN=F",
    "ZB":  "ZB=F",
    # FX
    "6E":  "6E=F",
    "6J":  "6J=F",
    "6B":  "6B=F",
    "6A":  "6A=F",
    # Micros (if ever needed)
    "MES": "MES=F",
    "MNQ": "MNQ=F",
    "MYM": "MYM=F",
    "M2K": "M2K=F",
}


class YFinanceDataProvider(FuturesDataProvider):
    """Fetch daily bars from Yahoo Finance using continuous futures tickers.

    Parameters
    ----------
    cache_dir
        Optional directory for caching downloaded data as parquet.
    """

    def __init__(
        self,
        *,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir else None

    def fetch_daily_bars(
        self, root: str, start: date, end: date,
    ) -> pd.DataFrame:
        """Fetch daily OHLCV bars for *root* from Yahoo Finance.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp, open, high, low, close, volume``
            (timestamp is UTC-aware).
        """
        # --- Check cache first ---
        if self.cache_dir is not None:
            cached = self._load_cached(root, start, end)
            if cached is not None:
                logger.info("YFinance cache hit: %s (%d rows)", root, len(cached))
                return cached

        # --- Map root to Yahoo ticker ---
        ticker = YFINANCE_TICKER_MAP.get(root)
        if ticker is None:
            logger.warning(
                "No Yahoo Finance ticker mapping for root '%s' — skipping", root,
            )
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

        # --- Download via yfinance ---
        try:
            import yfinance as yf

            logger.info("YFinance: downloading %s (%s) %s → %s", root, ticker, start, end)

            # yfinance end date is exclusive, so add 1 day
            yf_end = end + timedelta(days=1)

            raw = yf.download(
                ticker,
                start=start.isoformat(),
                end=yf_end.isoformat(),
                interval="1d",
                auto_adjust=True,
                progress=False,
            )

            if raw.empty:
                logger.warning("YFinance returned no data for %s (%s)", root, ticker)
                return pd.DataFrame(
                    columns=["timestamp", "open", "high", "low", "close", "volume"]
                )

            # Handle MultiIndex columns that yfinance sometimes returns
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            # Normalize to our schema
            df = pd.DataFrame({
                "timestamp": pd.to_datetime(raw.index, utc=True),
                "open": pd.to_numeric(raw["Open"], errors="coerce"),
                "high": pd.to_numeric(raw["High"], errors="coerce"),
                "low": pd.to_numeric(raw["Low"], errors="coerce"),
                "close": pd.to_numeric(raw["Close"], errors="coerce"),
                "volume": pd.to_numeric(raw["Volume"], errors="coerce").fillna(0).astype(int),
            })

            # Filter to requested range (safety check)
            mask = (df["timestamp"].dt.date >= start) & (df["timestamp"].dt.date <= end)
            df = df.loc[mask].reset_index(drop=True)

            # Drop any rows with NaN prices
            df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)

            # OHLC clamping: fix Yahoo data quality artifacts
            # Ensure high >= max(open, close) and low <= min(open, close)
            row_max = df[["open", "high", "low", "close"]].max(axis=1)
            row_min = df[["open", "high", "low", "close"]].min(axis=1)
            df["high"] = row_max
            df["low"] = row_min

            logger.info(
                "YFinance: %s → %d bars (%s → %s)",
                root, len(df),
                df["timestamp"].min().date() if not df.empty else "N/A",
                df["timestamp"].max().date() if not df.empty else "N/A",
            )

            # --- Cache ---
            if self.cache_dir is not None and not df.empty:
                self._save_cache(df, root, start, end)

            return df

        except Exception as e:
            logger.error("YFinance download failed for %s (%s): %s", root, ticker, e)
            return pd.DataFrame(
                columns=["timestamp", "open", "high", "low", "close", "volume"]
            )

    # -- Caching -------------------------------------------------------------

    def _cache_key(self, root: str, start: date, end: date) -> str:
        return f"yf_{root}_{start.isoformat()}_{end.isoformat()}"

    def _cache_path(self, root: str, start: date, end: date) -> Path:
        assert self.cache_dir is not None
        cache_subdir = self.cache_dir / "yfinance"
        cache_subdir.mkdir(parents=True, exist_ok=True)
        key = self._cache_key(root, start, end)
        return cache_subdir / f"{key}.parquet"

    def _load_cached(
        self, root: str, start: date, end: date,
    ) -> Optional[pd.DataFrame]:
        path = self._cache_path(root, start, end)
        if path.exists():
            df = pd.read_parquet(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
            return df
        return None

    def _save_cache(
        self, df: pd.DataFrame, root: str, start: date, end: date,
    ) -> None:
        path = self._cache_path(root, start, end)
        df.to_parquet(path, index=False)
        logger.info("Cached %d bars → %s", len(df), path)
