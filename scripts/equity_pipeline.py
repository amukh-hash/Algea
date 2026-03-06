"""
Sequence 2: SMoE Expert Data Pipeline

Downloads S&P 500 daily OHLCV (2018-present), computes rolling technical
features with cross-sectional Z-scoring, and partitions into 4 expert
datasets for the RankTransformer:

  Expert 0: Tech/Growth   (XLK, XLC, XLY sectors)
  Expert 1: Value/Defense (XLF, XLE, XLV, XLP, XLU sectors)
  Expert 2: Crisis        (All sectors, VIX_{t-1} > 20)
  Expert 3: Calm          (All sectors, VIX_{t-1} ≤ 20)

Output per expert:
  X_expert_{i}.npy  — [Num_Days, 512, Num_Features]  (Z-scored features)
  y_expert_{i}.npy  — [Num_Days, 512]                (rank targets 0→1)
  mask_expert_{i}.npy — [Num_Days, 512]              (True = padded/invalid)
"""
from __future__ import annotations

import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("SMoE_Pipeline")

MAX_SEQ_LEN = 512
START_DATE = "2018-01-01"
FWD_RETURN_DAYS = 3
OUTPUT_DIR = Path("data_lake/smoe_training")

# ═══════════════════════════════════════════════════════════════════════════
# GICS Sector → Expert Mapping
# ═══════════════════════════════════════════════════════════════════════════

# Expert 0: Tech/Growth (XLK, XLC, XLY)
EXPERT_0_SECTORS = {"Information Technology", "Communication Services", "Consumer Discretionary"}

# Expert 1: Value/Defensive (XLF, XLE, XLV, XLP, XLU)
EXPERT_1_SECTORS = {"Financials", "Energy", "Health Care",
                    "Consumer Staples", "Utilities"}

# Combined for reference:
# Industrials, Real Estate, Basic Materials → appear in Expert 2/3 (Crisis/Calm)
# but NOT in Expert 0 or 1 by sector. All sectors appear in Expert 2/3.


# ═══════════════════════════════════════════════════════════════════════════
# Step 1: Get S&P 500 Universe with Sectors
# ═══════════════════════════════════════════════════════════════════════════

def get_sp500_universe() -> pd.DataFrame:
    """Scrape current S&P 500 constituents from Wikipedia with GICS sectors."""
    import io
    import requests

    logger.info("Fetching S&P 500 constituents from Wikipedia...")
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(io.StringIO(resp.text))
    df = tables[0]
    # Standardize columns
    df = df.rename(columns={
        "Symbol": "ticker",
        "Security": "name",
        "GICS Sector": "sector",
        "GICS Sub-Industry": "sub_industry",
    })
    # Fix tickers with dots (e.g., BRK.B → BRK-B for yfinance)
    df["ticker"] = df["ticker"].str.replace(".", "-", regex=False)
    df = df[["ticker", "name", "sector", "sub_industry"]].copy()
    logger.info("S&P 500 universe: %d constituents", len(df))
    return df


# ═══════════════════════════════════════════════════════════════════════════
# Step 2: Download Daily OHLCV via yfinance
# ═══════════════════════════════════════════════════════════════════════════

def download_price_data(tickers: list[str], start: str = START_DATE) -> pd.DataFrame:
    """Batch download daily OHLCV for all tickers via yfinance."""
    logger.info("Downloading %d tickers from %s...", len(tickers), start)

    # Download in batches to avoid timeouts
    batch_size = 50
    all_frames = []

    for i in range(0, len(tickers), batch_size):
        batch = tickers[i:i + batch_size]
        logger.info("  Batch %d/%d (%d tickers)...",
                     i // batch_size + 1,
                     (len(tickers) + batch_size - 1) // batch_size,
                     len(batch))
        try:
            data = yf.download(
                batch,
                start=start,
                auto_adjust=True,
                group_by="ticker",
                threads=True,
                progress=False,
            )
            if data.empty:
                continue

            # Reshape from MultiIndex columns to long format
            for ticker in batch:
                try:
                    if len(batch) == 1:
                        ticker_data = data.copy()
                    else:
                        ticker_data = data[ticker].copy()

                    if ticker_data.empty or ticker_data["Close"].isna().all():
                        continue

                    ticker_df = pd.DataFrame({
                        "date": ticker_data.index,
                        "ticker": ticker,
                        "open": ticker_data["Open"].values,
                        "high": ticker_data["High"].values,
                        "low": ticker_data["Low"].values,
                        "close": ticker_data["Close"].values,
                        "volume": ticker_data["Volume"].values,
                    })
                    ticker_df = ticker_df.dropna(subset=["close"])
                    all_frames.append(ticker_df)
                except (KeyError, TypeError):
                    continue
        except Exception as e:
            logger.warning("  Batch download error: %s", e)
            continue

    if not all_frames:
        raise RuntimeError("No price data downloaded!")

    prices = pd.concat(all_frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"]).dt.date
    prices = prices.sort_values(["date", "ticker"]).reset_index(drop=True)

    n_tickers = prices["ticker"].nunique()
    n_days = prices["date"].nunique()
    logger.info("Downloaded: %d rows, %d tickers, %d trading days",
                len(prices), n_tickers, n_days)
    return prices


def download_vix(start: str = START_DATE) -> pd.DataFrame:
    """Download ^VIX daily close."""
    logger.info("Downloading ^VIX data...")
    vix = yf.download("^VIX", start=start, auto_adjust=True, progress=False)
    vix_df = pd.DataFrame({
        "date": vix.index,
        "vix_close": vix["Close"].values.flatten(),
    })
    vix_df["date"] = pd.to_datetime(vix_df["date"]).dt.date
    # Lag by 1 day: VIX_{t-1} determines regime for day t
    vix_df = vix_df.sort_values("date").reset_index(drop=True)
    vix_df["vix_prev"] = vix_df["vix_close"].shift(1)
    vix_df = vix_df.dropna(subset=["vix_prev"])
    logger.info("VIX data: %d days, regime split at 20.0", len(vix_df))
    return vix_df[["date", "vix_prev"]]


# ═══════════════════════════════════════════════════════════════════════════
# Step 3: Compute Rolling Technical Features
# ═══════════════════════════════════════════════════════════════════════════

def compute_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute per-stock rolling features. Returns the features joined to prices."""
    logger.info("Computing rolling technical features...")
    prices = prices.sort_values(["ticker", "date"]).copy()

    # Group by ticker for rolling computations
    g = prices.groupby("ticker")

    # --- Returns ---
    prices["ret_1d"] = g["close"].pct_change(1)
    prices["ret_5d"] = g["close"].pct_change(5)
    prices["ret_10d"] = g["close"].pct_change(10)
    prices["ret_20d"] = g["close"].pct_change(20)

    # --- Volatility ---
    prices["vol_10d"] = g["ret_1d"].transform(lambda x: x.rolling(10).std())
    prices["vol_21d"] = g["ret_1d"].transform(lambda x: x.rolling(21).std())

    # --- RSI (14-day) ---
    delta = g["close"].diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.groupby(prices["ticker"]).transform(lambda x: x.rolling(14).mean())
    avg_loss = loss.groupby(prices["ticker"]).transform(lambda x: x.rolling(14).mean())
    rs = avg_gain / avg_loss.replace(0, np.nan)
    prices["rsi_14"] = 100 - (100 / (1 + rs))

    # --- Volume Shock (current volume / 20d average volume) ---
    prices["vol_20d_avg"] = g["volume"].transform(lambda x: x.rolling(20).mean())
    prices["volume_shock"] = prices["volume"] / prices["vol_20d_avg"].replace(0, np.nan)

    # --- Forward 3-day Return (TARGET) ---
    prices["fwd_ret_3d"] = g["close"].pct_change(FWD_RETURN_DAYS).shift(-FWD_RETURN_DAYS)

    # Drop warmup period rows (first 21 trading days per stock have NaN features)
    prices = prices.dropna(subset=[
        "ret_5d", "ret_20d", "vol_10d", "vol_21d", "rsi_14", "volume_shock",
    ])

    # Drop rows without forward returns (last 3 days)
    prices = prices.dropna(subset=["fwd_ret_3d"])

    feature_cols = [
        "ret_1d", "ret_5d", "ret_10d", "ret_20d",
        "vol_10d", "vol_21d", "rsi_14", "volume_shock",
    ]
    logger.info(
        "Features computed: %d rows, %d features: %s",
        len(prices), len(feature_cols), feature_cols,
    )
    return prices, feature_cols


# ═══════════════════════════════════════════════════════════════════════════
# Step 4: Cross-Sectional Z-Scoring
# ═══════════════════════════════════════════════════════════════════════════

def cross_sectional_zscore(prices: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Z-score features ACROSS stocks for each day (cross-sectional)."""
    logger.info("Applying cross-sectional Z-scoring...")

    for col in feature_cols:
        mean = prices.groupby("date")[col].transform("mean")
        std = prices.groupby("date")[col].transform("std")
        prices[f"{col}_z"] = (prices[col] - mean) / std.replace(0, np.nan)

    # Replace any remaining NaN/inf with 0 (edge case: single stock on a day)
    z_cols = [f"{c}_z" for c in feature_cols]
    prices[z_cols] = prices[z_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    return prices, z_cols


# ═══════════════════════════════════════════════════════════════════════════
# Step 5: Cross-Sectional Rank Target
# ═══════════════════════════════════════════════════════════════════════════

def compute_rank_target(prices: pd.DataFrame) -> pd.DataFrame:
    """Rank forward 3-day returns cross-sectionally to [0, 1]."""
    logger.info("Computing cross-sectional rank targets...")
    prices["rank_target"] = prices.groupby("date")["fwd_ret_3d"].rank(pct=True)
    return prices


# ═══════════════════════════════════════════════════════════════════════════
# Step 6: Build Expert Partitions
# ═══════════════════════════════════════════════════════════════════════════

def build_expert_arrays(
    prices: pd.DataFrame,
    z_cols: list[str],
    sector_map: dict[str, str],
    vix_df: pd.DataFrame,
    output_dir: Path,
):
    """Build and save the 4 expert datasets."""
    logger.info("Building 4 expert datasets...")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Assign sectors
    prices["sector"] = prices["ticker"].map(sector_map).fillna("Unknown")

    # Merge VIX regime
    prices = prices.merge(vix_df, on="date", how="left")
    prices = prices.dropna(subset=["vix_prev"])

    # Define expert filters
    expert_filters = {
        0: ("Tech/Growth", prices["sector"].isin(EXPERT_0_SECTORS)),
        1: ("Value/Defensive", prices["sector"].isin(EXPERT_1_SECTORS)),
        2: ("Crisis (VIX>20)", pd.Series(True, index=prices.index) & (prices["vix_prev"] > 20.0)),
        3: ("Calm (VIX≤20)", pd.Series(True, index=prices.index) & (prices["vix_prev"] <= 20.0)),
    }

    n_features = len(z_cols)

    for expert_id, (name, mask) in expert_filters.items():
        logger.info("  Expert %d: %s", expert_id, name)
        subset = prices[mask].copy()

        if subset.empty:
            logger.warning("    EMPTY! Skipping.")
            continue

        # Re-rank targets within this expert's subset (cross-sectional within partition)
        subset["rank_target"] = subset.groupby("date")["fwd_ret_3d"].rank(pct=True)

        dates = sorted(subset["date"].unique())
        n_days = len(dates)
        logger.info("    %d trading days, %d total rows", n_days, len(subset))

        X = np.zeros((n_days, MAX_SEQ_LEN, n_features), dtype=np.float32)
        y = np.zeros((n_days, MAX_SEQ_LEN), dtype=np.float32)
        pad_mask = np.ones((n_days, MAX_SEQ_LEN), dtype=bool)  # True = padded

        for day_idx, date in enumerate(dates):
            day_data = subset[subset["date"] == date]
            n_stocks = min(len(day_data), MAX_SEQ_LEN)

            features = day_data[z_cols].values[:n_stocks]
            targets = day_data["rank_target"].values[:n_stocks]

            X[day_idx, :n_stocks, :] = features
            y[day_idx, :n_stocks] = targets
            pad_mask[day_idx, :n_stocks] = False  # Not padded

        # Summary stats
        active_counts = (~pad_mask).sum(axis=1)
        logger.info(
            "    Shape: X=%s, y=%s, mask=%s",
            X.shape, y.shape, pad_mask.shape,
        )
        logger.info(
            "    Active stocks/day: mean=%.0f, min=%d, max=%d",
            active_counts.mean(), active_counts.min(), active_counts.max(),
        )

        # Save
        np.save(output_dir / f"X_expert_{expert_id}.npy", X)
        np.save(output_dir / f"y_expert_{expert_id}.npy", y)
        np.save(output_dir / f"mask_expert_{expert_id}.npy", pad_mask)
        logger.info("    Saved to %s", output_dir)


# ═══════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("SEQUENCE 2: SMoE Expert Data Pipeline")
    logger.info("=" * 70)

    # 1. Universe
    universe = get_sp500_universe()
    tickers = universe["ticker"].tolist()
    sector_map = dict(zip(universe["ticker"], universe["sector"]))

    # Save universe for audit
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    universe.to_parquet(OUTPUT_DIR / "sp500_universe.parquet", index=False)

    # 2. Download price data
    cache_path = OUTPUT_DIR / "sp500_prices.parquet"
    if cache_path.exists():
        logger.info("Loading cached price data from %s", cache_path)
        prices = pd.read_parquet(cache_path)
    else:
        prices = download_price_data(tickers, start=START_DATE)
        prices.to_parquet(cache_path, index=False)
        logger.info("Cached price data to %s", cache_path)

    # 3. Download VIX
    vix_df = download_vix(start=START_DATE)

    # 4. Compute features
    prices, feature_cols = compute_features(prices)

    # 5. Cross-sectional Z-scoring
    prices, z_cols = cross_sectional_zscore(prices, feature_cols)

    # 6. Cross-sectional rank target
    prices = compute_rank_target(prices)

    # 7. Build expert partitions
    build_expert_arrays(prices, z_cols, sector_map, vix_df, OUTPUT_DIR)

    logger.info("=" * 70)
    logger.info("SEQUENCE 2 DATA PIPELINE COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
