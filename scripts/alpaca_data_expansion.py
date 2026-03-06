"""
Alpaca Data Expansion Pipeline

Phase 1: Download 4 years of 5-min bars for 13 ETFs (iTransformer Seq 3R)
         XLF, XLK, XLE, XLI, XLY, XLP, XLV, XLU, XLB, XLRE, XLC, SPY, QQQ
         2022-01-01 → Present → ~78K samples per ETF

Phase 2: Download Russell 3000 daily bars for SMoE universe expansion (Seq 2.5R)
         2018-01-01 → Present → ~3000 tickers × 2000 days

Uses Alpaca v2 Market Data API (Free Tier: 200 req/min).
"""
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AlpacaExpansion")

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
BASE_URL = "https://data.alpaca.markets/v2"
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET_KEY}

# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

# Phase 1: iTransformer ETFs (5-min bars, 4 years)
ITRANS_ETFS = [
    "XLF", "XLK", "XLE", "XLI", "XLY", "XLP",
    "XLV", "XLU", "XLB", "XLRE", "XLC",
    "SPY", "QQQ",
]
ITRANS_START = "2022-01-01"
ITRANS_TIMEFRAME = "5Min"

# Output directories
ITRANS_OUTPUT = Path("data_lake/statarb_expanded")
RUSSELL_OUTPUT = Path("data_lake/russell3000_daily")


# ═══════════════════════════════════════════════════════════════════════════
# Async Downloader
# ═══════════════════════════════════════════════════════════════════════════

async def fetch_all_bars(
    client: httpx.AsyncClient,
    symbol: str,
    start: str,
    end: str,
    timeframe: str = "5Min",
) -> list[dict]:
    """Fetch all bars with pagination from Alpaca v2 API."""
    url = f"{BASE_URL}/stocks/{symbol}/bars"
    all_bars = []
    page_token = None

    params = {
        "start": f"{start}T00:00:00Z",
        "end": f"{end}T23:59:59Z",
        "timeframe": timeframe,
        "limit": 10000,
        "adjustment": "split",
    }

    while True:
        if page_token:
            params["page_token"] = page_token

        try:
            resp = await client.get(url, params=params, headers=HEADERS, timeout=30.0)
            if resp.status_code == 429:
                logger.warning("  Rate limited on %s, sleeping 5s...", symbol)
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

            await asyncio.sleep(0.3)  # Rate limit politeness

        except Exception as e:
            logger.error("  Exception on %s: %s", symbol, e)
            await asyncio.sleep(2)
            break

    return all_bars


def bars_to_dataframe(bars: list[dict], symbol: str) -> pd.DataFrame:
    """Convert Alpaca bar list to a clean DataFrame."""
    if not bars:
        return pd.DataFrame()

    df = pd.DataFrame(bars)
    df = df.rename(columns={
        "t": "timestamp", "o": "open", "h": "high",
        "l": "low", "c": "close", "v": "volume",
        "n": "trade_count", "vw": "vwap",
    })
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["symbol"] = symbol
    df = df.set_index("timestamp").sort_index()
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Phase 1: iTransformer ETF Download (5-min, 4 years)
# ═══════════════════════════════════════════════════════════════════════════

async def download_itransformer_etfs():
    """Download 4 years of 5-min bars for 13 ETFs."""
    logger.info("=" * 60)
    logger.info("PHASE 1: iTransformer ETF Data (5-min bars, 4 years)")
    logger.info("  Tickers: %s", ITRANS_ETFS)
    logger.info("  Period: %s → present", ITRANS_START)
    logger.info("=" * 60)

    ITRANS_OUTPUT.mkdir(parents=True, exist_ok=True)
    end_date = datetime.now().strftime("%Y-%m-%d")

    async with httpx.AsyncClient(
        limits=httpx.Limits(max_keepalive_connections=5, max_connections=5),
    ) as client:
        all_dfs = {}

        for symbol in ITRANS_ETFS:
            cache_path = ITRANS_OUTPUT / f"{symbol}_5min.parquet"
            if cache_path.exists():
                logger.info("  %s: cached, loading from disk", symbol)
                all_dfs[symbol] = pd.read_parquet(cache_path)
                continue

            logger.info("  Downloading %s...", symbol)
            bars = await fetch_all_bars(client, symbol, ITRANS_START, end_date, ITRANS_TIMEFRAME)
            df = bars_to_dataframe(bars, symbol)

            if df.empty:
                logger.warning("  %s: NO DATA", symbol)
                continue

            df.to_parquet(cache_path)
            all_dfs[symbol] = df
            logger.info("  %s: %d bars (%.1f MB)", symbol, len(df),
                        cache_path.stat().st_size / 1024 / 1024)
            await asyncio.sleep(1)  # Cool down

    # Build aligned multivariate matrix
    logger.info("Building aligned 5-min price matrix...")
    close_frames = {}
    for sym, df in all_dfs.items():
        close_frames[sym] = df["close"].rename(sym)

    if close_frames:
        aligned = pd.concat(close_frames.values(), axis=1, join="inner")
        aligned = aligned.dropna()
        aligned.to_parquet(ITRANS_OUTPUT / "aligned_5min_prices.parquet")
        logger.info("Aligned matrix: %s (%d bars × %d ETFs)",
                    aligned.shape, len(aligned), aligned.shape[1])

        # Build features: returns (pct_change over lookback window)
        build_itransformer_features(aligned)
    else:
        logger.error("No data downloaded!")


def build_itransformer_features(aligned: pd.DataFrame):
    """Build iTransformer X and y arrays from aligned 5-min prices."""
    logger.info("Computing iTransformer features and targets...")

    returns = aligned.pct_change().dropna()
    ret_values = returns.values
    n_bars, n_variates = ret_values.shape
    lookback = 60   # 60 × 5min = 5 hours of lookback
    fwd_horizon = 12  # 12 × 5min = 1 hour forward

    # Rolling Z-score per variate (relative value indicator)
    rolling_z = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for col in returns.columns:
        rmean = returns[col].rolling(lookback).mean()
        rstd = returns[col].rolling(lookback).std().replace(0, np.nan)
        rolling_z[col] = (returns[col] - rmean) / rstd

    # Target: Δ Z-score forward = Z_{t+12} - Z_t
    rz_values = rolling_z.values

    # Valid range: need lookback warmup at start AND fwd_horizon room at end
    # Sample i uses returns[i : i+lookback] as features
    # and rz_values[i+lookback+fwd_horizon] - rz_values[i+lookback] as target
    max_i = n_bars - lookback - fwd_horizon
    min_i = lookback  # Ensure rolling Z has warmed up

    if max_i <= min_i:
        logger.error("Not enough data for sliding windows!")
        return

    n_samples = max_i - min_i
    X = np.zeros((n_samples, lookback, n_variates), dtype=np.float32)
    y = np.zeros((n_samples, n_variates), dtype=np.float32)

    for i in range(n_samples):
        idx = min_i + i
        X[i] = ret_values[idx:idx + lookback]
        z_t = rz_values[idx + lookback]
        z_fwd = rz_values[idx + lookback + fwd_horizon]
        y[i] = z_fwd - z_t

    # Remove any NaN targets (from rolling warmup edge)
    valid_mask = ~np.any(np.isnan(y), axis=1)
    X = X[valid_mask]
    y = y[valid_mask]

    # Clip extreme outliers
    y = np.clip(y, -5.0, 5.0)

    logger.info("iTransformer arrays: X=%s y=%s", X.shape, y.shape)
    logger.info("  y stats: mean=%.4f std=%.4f range=[%.2f, %.2f]",
                y.mean(), y.std(), y.min(), y.max())

    np.save(ITRANS_OUTPUT / "X_features.npy", X)
    np.save(ITRANS_OUTPUT / "y_targets.npy", y)
    logger.info("Saved to %s", ITRANS_OUTPUT)


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    if not API_KEY or not SECRET_KEY:
        logger.error("ALPACA_API_KEY / ALPACA_SECRET_KEY not found in .env")
        sys.exit(1)

    logger.info("Alpaca Data Expansion Pipeline")
    logger.info("  API Key: %s...%s", API_KEY[:6], API_KEY[-4:])

    await download_itransformer_etfs()

    logger.info("=" * 60)
    logger.info("DATA EXPANSION COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
