"""
StatArb Bar Data Pipeline — ETF sector ingestion + feature engineering.

Fetches, aligns, and transforms 5-minute bar data for the iTransformer
multivariate statistical arbitrage model.

Features:
    - Historical ingestion via yfinance (fallback: Databento/IBKR)
    - Forward-fill alignment (Zero-Order Hold) across variates
    - Rolling log-returns + cointegration spread z-scores
    - Output: [Batch, Time, Variates] tensors ready for iTransformer
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Canonical StatArb universe
STATARB_UNIVERSE = ["XLF", "XLK", "XLE", "XLI", "XLY", "XLP"]


# ═══════════════════════════════════════════════════════════════════════════
# Data Ingestion
# ═══════════════════════════════════════════════════════════════════════════

def fetch_sector_etf_bars(
    symbols: list[str] = STATARB_UNIVERSE,
    interval: str = "5m",
    lookback_days: int = 60,
    cache_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch intraday bars for sector ETFs.

    Parameters
    ----------
    symbols : list[str]
        Ticker symbols to fetch.
    interval : str
        Bar interval (e.g., "5m", "15m", "1d").
    lookback_days : int
        Number of calendar days of history to fetch.
        Note: yfinance limits 5m bars to ~60 days.
    cache_dir : str or None
        If provided, caches raw data as parquet.

    Returns
    -------
    pd.DataFrame
        MultiIndex DataFrame with columns per symbol and OHLCV fields.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError(
            "yfinance is required for StatArb data ingestion.  "
            "pip install yfinance"
        )

    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)

    frames: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=interval,
                auto_adjust=True,
            )
            if df.empty:
                logger.warning("No data returned for %s", sym)
                continue
            # Keep only Close for simplicity
            frames[sym] = df[["Close"]].rename(columns={"Close": sym})
            logger.info("Fetched %d bars for %s", len(df), sym)
        except Exception as e:
            logger.error("Failed to fetch %s: %s", sym, e)

    if not frames:
        raise RuntimeError("No data fetched for any symbol")

    # Merge on timestamp index
    combined = pd.concat(frames.values(), axis=1, join="outer")

    if cache_dir:
        cache_path = Path(cache_dir) / "statarb_raw_bars.parquet"
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_parquet(cache_path)
        logger.info("Cached raw bars to %s", cache_path)

    return combined


# ═══════════════════════════════════════════════════════════════════════════
# Alignment Engine — Zero-Order Hold
# ═══════════════════════════════════════════════════════════════════════════

def align_forward_fill(df: pd.DataFrame) -> pd.DataFrame:
    """Apply strict Zero-Order Hold forward-fill across all columns.

    Critical: timestamp misalignment between variates artificially inflates
    spread volatility, destroying the iTransformer's attention matrix.

    Parameters
    ----------
    df : pd.DataFrame
        Multivariate price DataFrame with potential NaN gaps.

    Returns
    -------
    pd.DataFrame
        Fully filled DataFrame with no NaN values.
    """
    # Forward-fill first (carry last known close)
    aligned = df.ffill()
    # backward-fill the very first rows  if a symbol starts trading later
    aligned = aligned.bfill()

    nan_count = aligned.isna().sum().sum()
    if nan_count > 0:
        logger.warning(
            "After alignment, %d NaN values remain.  "
            "Dropping rows with NaN.", nan_count
        )
        aligned = aligned.dropna()

    return aligned


# ═══════════════════════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════

def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute rolling log-returns from price DataFrame.

    Returns
    -------
    pd.DataFrame
        Log-returns with the first row dropped (NaN from diff).
    """
    return np.log(prices / prices.shift(1)).dropna()


def compute_spread_zscores(
    prices: pd.DataFrame,
    lookback: int = 60,
) -> pd.DataFrame:
    """Compute pairwise cointegration spread z-scores.

    For each pair (i, j) where i < j, compute:
        spread = price_i - β * price_j
    where β is the OLS regression coefficient over the lookback window.
    Then normalize to a z-score.

    Parameters
    ----------
    prices : pd.DataFrame
        Aligned price DataFrame.
    lookback : int
        Rolling window for OLS and z-score computation.

    Returns
    -------
    pd.DataFrame
        DataFrame of z-scores, one column per pair.
    """
    symbols = prices.columns.tolist()
    spreads: dict[str, pd.Series] = {}

    for i, sym_a in enumerate(symbols):
        for sym_b in symbols[i + 1:]:
            pair_name = f"{sym_a}_{sym_b}_zscore"
            # Rolling OLS approximation: β ≈ cov(a,b) / var(b)
            rolling_cov = prices[sym_a].rolling(lookback).cov(prices[sym_b])
            rolling_var = prices[sym_b].rolling(lookback).var()
            beta = rolling_cov / (rolling_var + 1e-10)

            spread = prices[sym_a] - beta * prices[sym_b]
            spread_mean = spread.rolling(lookback).mean()
            spread_std = spread.rolling(lookback).std() + 1e-10
            zscore = (spread - spread_mean) / spread_std

            spreads[pair_name] = zscore

    return pd.DataFrame(spreads).dropna()


def build_itransformer_features(
    prices: pd.DataFrame,
    lookback_len: int = 60,
) -> np.ndarray:
    """Build feature tensor for iTransformer from aligned prices.

    Combines log-returns and spread z-scores into a single
    multivariate panel.

    Parameters
    ----------
    prices : pd.DataFrame
        Aligned price DataFrame with columns = variate names.
    lookback_len : int
        Window length for each sample.

    Returns
    -------
    np.ndarray
        Shape ``[num_samples, lookback_len, num_variates]``.
        Variates = individual ETF returns (no pair features in the
        iTransformer — those are learned via cross-variate attention).
    """
    log_rets = compute_log_returns(prices)

    # Convert to numpy: [Time, Variates]
    data = log_rets.values.astype(np.float32)
    T, N = data.shape

    if T < lookback_len:
        raise ValueError(
            f"Not enough data: got {T} timesteps, need >= {lookback_len}"
        )

    # Sliding window: [num_samples, lookback_len, num_variates]
    num_samples = T - lookback_len + 1
    windows = np.lib.stride_tricks.sliding_window_view(data, lookback_len, axis=0)
    # windows shape: [num_samples, N, lookback_len] → transpose to [num_samples, lookback_len, N]
    windows = windows.transpose(0, 2, 1)

    logger.info(
        "Built iTransformer features: %d samples × %d timesteps × %d variates",
        windows.shape[0], windows.shape[1], windows.shape[2],
    )

    return windows


# ═══════════════════════════════════════════════════════════════════════════
# Full Pipeline
# ═══════════════════════════════════════════════════════════════════════════

def run_statarb_pipeline(
    symbols: list[str] = STATARB_UNIVERSE,
    interval: str = "5m",
    lookback_days: int = 59,
    lookback_len: int = 60,
    output_dir: str = "data_lake/statarb",
) -> dict:
    """End-to-end StatArb data pipeline.

    1. Fetch raw bars
    2. Align via forward-fill
    3. Compute features
    4. Save to disk

    Returns
    -------
    dict
        Pipeline summary with paths and shapes.
    """
    logger.info("Starting StatArb pipeline for %s", symbols)

    # 1. Fetch
    raw = fetch_sector_etf_bars(
        symbols=symbols,
        interval=interval,
        lookback_days=lookback_days,
        cache_dir=output_dir,
    )

    # 2. Align
    aligned = align_forward_fill(raw)
    logger.info("Aligned prices: %d rows × %d columns", *aligned.shape)

    # 3. Features
    features = build_itransformer_features(aligned, lookback_len=lookback_len)

    # Also compute spread z-scores for cointegration monitoring
    zscores = compute_spread_zscores(aligned, lookback=lookback_len)

    # 4. Save
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    features_path = out_path / "itransformer_features.npy"
    np.save(features_path, features)

    zscores_path = out_path / "spread_zscores.parquet"
    zscores.to_parquet(zscores_path)

    aligned_path = out_path / "aligned_prices.parquet"
    aligned.to_parquet(aligned_path)

    summary = {
        "status": "ok",
        "num_symbols": len(symbols),
        "features_shape": list(features.shape),
        "features_path": str(features_path),
        "zscores_path": str(zscores_path),
        "aligned_prices_path": str(aligned_path),
        "date_range": [
            str(aligned.index.min()),
            str(aligned.index.max()),
        ],
    }
    logger.info("StatArb pipeline complete: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_statarb_pipeline()
    print(result)
