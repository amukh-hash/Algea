"""
Alpaca 60-Minute ETF Data Expansion for iTransformer DA Improvement

Pulls 8 years of 60-min bars for 13 ETFs to step over the HFT microwave
boundary. At 60-min resolution:
- lookback=60 → 8.5 trading days
- 12-bar target → 1.7 trading days (institutional VWAP rotation)
- Expected ~6K-8K bars/ETF/year × 8 years = ~50-60K samples

Alpaca free tier limits historical intraday to ~5 years, so we'll get
what's available (2020-01-01 to present) and adjust.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Alpaca60min")

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://data.alpaca.markets/v2"
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY}

ETFS = [
    "XLF", "XLK", "XLE", "XLI", "XLY", "XLP",
    "XLV", "XLU", "XLB", "XLRE", "XLC",
    "SPY", "QQQ",
]
OUTPUT = Path("data_lake/statarb_60min")


async def fetch_all_bars(client, symbol, start, end, timeframe="1Hour"):
    url = f"{BASE_URL}/stocks/{symbol}/bars"
    all_bars = []
    page_token = None
    params = {
        "start": f"{start}T00:00:00Z", "end": f"{end}T23:59:59Z",
        "timeframe": timeframe, "limit": 10000, "adjustment": "split",
    }
    while True:
        if page_token:
            params["page_token"] = page_token
        try:
            resp = await client.get(url, params=params, headers=HEADERS, timeout=30.0)
            if resp.status_code == 429:
                await asyncio.sleep(5)
                continue
            if resp.status_code != 200:
                logger.error("  Error %d on %s: %s", resp.status_code, symbol, resp.text[:200])
                break
            data = resp.json()
            bars = data.get("bars", [])
            if not bars:
                break
            all_bars.extend(bars)
            page_token = data.get("next_page_token")
            if not page_token:
                break
            await asyncio.sleep(0.3)
        except Exception as e:
            logger.error("  Exception on %s: %s", symbol, e)
            break
    return all_bars


async def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    end_date = "2026-03-03"
    start_date = "2020-01-01"  # Alpaca free tier max ~5-6 years for intraday

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=5),
    ) as client:
        all_dfs = {}
        for sym in ETFS:
            cache = OUTPUT / f"{sym}_60min.parquet"
            if cache.exists():
                logger.info("  %s: cached", sym)
                all_dfs[sym] = pd.read_parquet(cache)
                continue

            logger.info("  Downloading %s (60-min, %s → %s)...", sym, start_date, end_date)
            bars = await fetch_all_bars(client, sym, start_date, end_date, "1Hour")
            if not bars:
                logger.warning("  %s: NO DATA", sym)
                continue

            df = pd.DataFrame(bars).rename(columns={
                "t": "timestamp", "o": "open", "h": "high",
                "l": "low", "c": "close", "v": "volume",
                "n": "trade_count", "vw": "vwap",
            })
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df["symbol"] = sym
            df = df.set_index("timestamp").sort_index()
            df.to_parquet(cache)
            all_dfs[sym] = df
            logger.info("  %s: %d bars (%.1f MB)", sym, len(df),
                        cache.stat().st_size / 1024 / 1024)
            await asyncio.sleep(1)

    # Build aligned matrix
    logger.info("Building aligned 60-min price matrix...")
    close_frames = {sym: df["close"].rename(sym) for sym, df in all_dfs.items()}
    aligned = pd.concat(close_frames.values(), axis=1, join="inner").dropna()
    aligned.to_parquet(OUTPUT / "aligned_60min_prices.parquet")
    logger.info("Aligned: %s (%d bars × %d ETFs)", aligned.shape, len(aligned), aligned.shape[1])

    # Build features
    build_features(aligned)


def build_features(aligned):
    logger.info("Computing 60-min iTransformer features and targets...")
    returns = aligned.pct_change().dropna()
    ret_values = returns.values
    n_bars, n_variates = ret_values.shape
    lookback = 60
    fwd_horizon = 12

    # Rolling Z-score
    rolling_z = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for col in returns.columns:
        rmean = returns[col].rolling(lookback).mean()
        rstd = returns[col].rolling(lookback).std().replace(0, np.nan)
        rolling_z[col] = (returns[col] - rmean) / rstd
    rz_values = rolling_z.values

    max_i = n_bars - lookback - fwd_horizon
    min_i = lookback
    if max_i <= min_i:
        logger.error("Not enough data!")
        return

    n_samples = max_i - min_i
    X = np.zeros((n_samples, lookback, n_variates), dtype=np.float32)
    y = np.zeros((n_samples, n_variates), dtype=np.float32)

    for i in range(n_samples):
        idx = min_i + i
        X[i] = ret_values[idx:idx + lookback]
        y[i] = rz_values[idx + lookback + fwd_horizon] - rz_values[idx + lookback]

    valid = ~np.any(np.isnan(y), axis=1)
    X, y = X[valid], y[valid]
    y = np.clip(y, -5.0, 5.0)

    logger.info("60-min iTransformer: X=%s y=%s", X.shape, y.shape)
    logger.info("  y stats: mean=%.4f std=%.4f range=[%.2f, %.2f]",
                y.mean(), y.std(), y.min(), y.max())

    np.save(OUTPUT / "X_features_60min.npy", X)
    np.save(OUTPUT / "y_targets_60min.npy", y)
    logger.info("Saved to %s", OUTPUT)


if __name__ == "__main__":
    asyncio.run(main())
