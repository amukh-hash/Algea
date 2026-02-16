"""Daily canonical futures data updater.

Downloads / refreshes daily bars through yesterday via yfinance and
writes a canonical parquet file for use by the paper-trading runner.

Usage::

    python backend/scripts/data/update_futures_daily.py
    python backend/scripts/data/update_futures_daily.py --output my_cache.parquet
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sleeves.cooc_reversal_futures.contract_master import CONTRACT_MASTER
from sleeves.cooc_reversal_futures.roll import active_contract_for_day

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# yfinance ticker mapping (continuous futures proxies)
# ---------------------------------------------------------------------------

_YF_TICKER_MAP = {
    "ES": "ES=F",
    "NQ": "NQ=F",
    "RTY": "RTY=F",
    "YM": "YM=F",
}

_DEFAULT_OUTPUT = "data_cache/canonical_futures_daily.parquet"
_DEFAULT_ROOTS = ["ES", "NQ", "RTY", "YM"]
_DEFAULT_START = "2015-01-02"


def _fetch_daily_bars(
    roots: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Fetch daily OHLCV bars from yfinance for all roots."""
    import yfinance as yf  # type: ignore[import-untyped]

    frames: list[pd.DataFrame] = []
    for root in roots:
        yf_ticker = _YF_TICKER_MAP.get(root)
        if not yf_ticker:
            logger.warning("No yfinance ticker for %s — skipping", root)
            continue

        logger.info("Downloading %s (%s) %s→%s", root, yf_ticker, start, end)
        ticker = yf.Ticker(yf_ticker)
        hist = ticker.history(start=start, end=end, interval="1d")

        if hist.empty:
            logger.warning("No data returned for %s", root)
            continue

        # Normalize
        hist = hist.reset_index()

        # Handle timezone: yfinance returns ET-localized timestamps
        date_col = "Date" if "Date" in hist.columns else hist.columns[0]
        hist["trading_day"] = pd.to_datetime(hist[date_col]).dt.tz_localize(None).dt.normalize()

        hist = hist.rename(columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        })

        # Select columns
        cols = ["trading_day", "open", "high", "low", "close", "volume"]
        hist = hist[[c for c in cols if c in hist.columns]].copy()

        # Drop NaN rows
        hist = hist.dropna(subset=["open", "high", "low", "close"])

        # OHLC clamp (deterministic fix for data quality)
        hist["low"] = hist[["low", "open", "close"]].min(axis=1)
        hist["high"] = hist[["high", "open", "close"]].max(axis=1)

        hist["root"] = root
        hist = hist.sort_values("trading_day").drop_duplicates(subset=["trading_day"], keep="last")

        # Compute returns
        hist["ret_co"] = hist["close"].pct_change() - hist["open"].pct_change()
        hist["ret_oc"] = (hist["close"] / hist["open"]) - 1.0

        # Active contract mapping
        active_contracts = []
        for _, row in hist.iterrows():
            day = row["trading_day"].date()
            spec = CONTRACT_MASTER[root]
            active_contracts.append(active_contract_for_day(root, day, spec))
        hist["active_contract"] = active_contracts

        # Rolling features (used by sleeve)
        hist["rolling_std_ret_co"] = hist["ret_co"].rolling(20, min_periods=5).std()

        # Metadata columns
        hist["provider_name"] = "yfinance"
        hist["provider_type"] = "CONTINUOUS_SYNTHETIC"
        hist["session_mode"] = "UNKNOWN"

        frames.append(hist)
        logger.info("  %s: %d bars", root, len(hist))

    if not frames:
        raise RuntimeError("No data fetched for any root")

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["trading_day", "root"]).reset_index(drop=True)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Update canonical futures daily data")
    parser.add_argument("--output", default=_DEFAULT_OUTPUT, help="Output parquet path")
    parser.add_argument("--roots", nargs="+", default=_DEFAULT_ROOTS, help="Root symbols")
    parser.add_argument("--start", default=_DEFAULT_START, help="Start date YYYY-MM-DD")
    args = parser.parse_args()

    yesterday = (date.today() - timedelta(days=1)).isoformat()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = _fetch_daily_bars(args.roots, args.start, yesterday)
    df.to_parquet(output_path, index=False)
    logger.info("Wrote %d rows to %s", len(df), output_path)


if __name__ == "__main__":
    main()
