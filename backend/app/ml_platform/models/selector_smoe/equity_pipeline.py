"""
Equity Feature Extraction Pipeline — real market data for SMoE training.

Fetches daily OHLCV from yfinance for a diversified equity universe and
computes 18 standard quantitative features per stock per day.

Feature Vector (18-dim):
    Returns:      ret_1d, ret_5d, ret_10d, ret_20d
    Volatility:   vol_5d, vol_10d, vol_20d, vol_60d
    Momentum:     mom_20d, mom_60d, rsi_14d
    Volume:       dollar_vol, volume_ratio_5d
    Mean-Rev:     zscore_20d, zscore_60d
    Risk:         max_drawdown_20d, beta_20d, corr_spy_20d

Output: features.npy [days, stocks, 18] + returns.npy [days, stocks]
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Diversified equity universe (sector representatives + high-liquidity names)
DEFAULT_UNIVERSE = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "NVDA", "META",
    # Financials
    "JPM", "BAC", "GS", "WFC",
    # Healthcare
    "JNJ", "UNH", "PFE",
    # Consumer
    "AMZN", "TSLA", "HD",
    # Energy
    "XOM", "CVX",
    # Industrials
    "CAT", "BA",
    # Benchmark
    "SPY",
]


def fetch_equity_data(
    symbols: list[str] = DEFAULT_UNIVERSE,
    period: str = "2y",
    interval: str = "1d",
) -> dict[str, pd.DataFrame]:
    """Fetch daily OHLCV for all symbols.

    Returns
    -------
    dict[str, pd.DataFrame]
        Keyed by symbol, each DataFrame has standard OHLCV columns.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required: pip install yfinance")

    data = {}
    for sym in symbols:
        try:
            ticker = yf.Ticker(sym)
            df = ticker.history(period=period, interval=interval, auto_adjust=True)
            if df.empty:
                logger.warning("No data for %s", sym)
                continue
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.columns = ["open", "high", "low", "close", "volume"]
            data[sym] = df
            logger.info("Fetched %d bars for %s", len(df), sym)
        except Exception as e:
            logger.error("Failed to fetch %s: %s", sym, e)

    return data


def compute_features(price_data: dict[str, pd.DataFrame]) -> tuple[np.ndarray, np.ndarray, list[str], list]:
    """Compute 18-dimensional feature vector for each stock×day.

    Parameters
    ----------
    price_data : dict
        Keyed by symbol, values are OHLCV DataFrames.

    Returns
    -------
    features : np.ndarray
        Shape ``[num_days, num_stocks, 18]``.
    returns : np.ndarray
        Shape ``[num_days, num_stocks]`` — next-day forward returns.
    symbols : list[str]
        Ordered list of stock symbols.
    dates : list
        Ordered list of trading dates.
    """
    symbols = sorted(price_data.keys())

    # Find the common date range
    all_dates = None
    for sym in symbols:
        dates = price_data[sym].index
        if all_dates is None:
            all_dates = set(dates)
        else:
            all_dates = all_dates.intersection(dates)

    dates_sorted = sorted(all_dates)
    if len(dates_sorted) < 80:
        raise ValueError(f"Need >= 80 days of overlapping data, got {len(dates_sorted)}")

    logger.info("Common dates: %d (from %s to %s)", len(dates_sorted), dates_sorted[0], dates_sorted[-1])

    # Get SPY for beta/correlation computation
    spy_close = price_data.get("SPY")
    if spy_close is not None:
        spy_returns = spy_close["close"].pct_change()
    else:
        spy_returns = None

    # Build feature matrix
    n_days = len(dates_sorted)
    n_stocks = len(symbols)
    n_features = 18

    features = np.full((n_days, n_stocks, n_features), np.nan, dtype=np.float32)
    fwd_returns = np.full((n_days, n_stocks), np.nan, dtype=np.float32)

    for s_idx, sym in enumerate(symbols):
        df = price_data[sym].loc[dates_sorted].copy()
        close = df["close"]
        volume = df["volume"]
        high = df["high"]
        low = df["low"]

        # Returns
        ret_1d = close.pct_change(1)
        ret_5d = close.pct_change(5)
        ret_10d = close.pct_change(10)
        ret_20d = close.pct_change(20)

        # Volatility (annualized)
        vol_5d = ret_1d.rolling(5).std() * np.sqrt(252)
        vol_10d = ret_1d.rolling(10).std() * np.sqrt(252)
        vol_20d = ret_1d.rolling(20).std() * np.sqrt(252)
        vol_60d = ret_1d.rolling(60).std() * np.sqrt(252)

        # Momentum
        mom_20d = close / close.shift(20) - 1
        mom_60d = close / close.shift(60) - 1

        # RSI 14
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi_14d = 100 - 100 / (1 + rs)
        rsi_14d = (rsi_14d - 50) / 50  # Normalize to [-1, 1]

        # Dollar volume
        dollar_vol = close * volume
        dollar_vol_norm = np.log1p(dollar_vol) / 20  # Normalize

        # Volume ratio
        vol_sma5 = volume.rolling(5).mean()
        vol_sma20 = volume.rolling(20).mean()
        volume_ratio = vol_sma5 / (vol_sma20 + 1)

        # Z-score (mean reversion)
        sma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        zscore_20d = (close - sma20) / (std20 + 1e-10)

        sma60 = close.rolling(60).mean()
        std60 = close.rolling(60).std()
        zscore_60d = (close - sma60) / (std60 + 1e-10)

        # Max drawdown 20d
        rolling_max = close.rolling(20).max()
        drawdown = (close - rolling_max) / (rolling_max + 1e-10)

        # Beta and correlation to SPY
        if spy_returns is not None:
            spy_ret_aligned = spy_returns.reindex(dates_sorted)
            stock_ret = ret_1d
            rolling_cov = stock_ret.rolling(20).cov(spy_ret_aligned)
            rolling_var = spy_ret_aligned.rolling(20).var()
            beta_20d = rolling_cov / (rolling_var + 1e-10)
            corr_spy = stock_ret.rolling(20).corr(spy_ret_aligned)
        else:
            beta_20d = pd.Series(1.0, index=dates_sorted)
            corr_spy = pd.Series(0.0, index=dates_sorted)

        # Forward 1-day return (target)
        fwd_ret = close.pct_change(1).shift(-1)

        # Stack features: [ret_1d, ret_5d, ret_10d, ret_20d,
        #                   vol_5d, vol_10d, vol_20d, vol_60d,
        #                   mom_20d, mom_60d, rsi_14d,
        #                   dollar_vol, volume_ratio,
        #                   zscore_20d, zscore_60d,
        #                   drawdown, beta, corr_spy]
        feat_stack = np.column_stack([
            ret_1d.values, ret_5d.values, ret_10d.values, ret_20d.values,
            vol_5d.values, vol_10d.values, vol_20d.values, vol_60d.values,
            mom_20d.values, mom_60d.values, rsi_14d.values,
            dollar_vol_norm.values, volume_ratio.values,
            zscore_20d.values, zscore_60d.values,
            drawdown.values, beta_20d.values, corr_spy.values,
        ])

        features[:, s_idx, :] = feat_stack
        fwd_returns[:, s_idx] = fwd_ret.values

    # Drop rows with any NaN (due to rolling windows — typically first 60 rows)
    valid_mask = ~np.any(np.isnan(features), axis=(1, 2))
    valid_mask &= ~np.any(np.isnan(fwd_returns), axis=1)
    features = features[valid_mask]
    fwd_returns = fwd_returns[valid_mask]
    valid_dates = [d for d, v in zip(dates_sorted, valid_mask) if v]

    logger.info(
        "Feature matrix: %d days × %d stocks × %d features (dropped %d warmup rows)",
        features.shape[0], features.shape[1], features.shape[2],
        n_days - features.shape[0],
    )

    # Replace any remaining NaN with 0
    features = np.nan_to_num(features, nan=0.0)
    fwd_returns = np.nan_to_num(fwd_returns, nan=0.0)

    return features, fwd_returns, symbols, valid_dates


def run_equity_pipeline(
    symbols: list[str] = DEFAULT_UNIVERSE,
    period: str = "2y",
    output_dir: str = "data_lake/smoe_features",
) -> dict:
    """End-to-end equity feature extraction.

    Returns
    -------
    dict
        Pipeline summary with paths and shapes.
    """
    logger.info("Starting equity feature extraction for %d symbols", len(symbols))

    # 1. Fetch
    raw_data = fetch_equity_data(symbols, period=period)

    # 2. Compute features
    features, returns, final_syms, dates = compute_features(raw_data)

    # 3. Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "equity_features.npy", features)
    np.save(out / "equity_returns.npy", returns)

    # Save symbol list
    with open(out / "symbols.txt", "w") as f:
        f.write("\n".join(final_syms))

    summary = {
        "status": "ok",
        "features_shape": list(features.shape),
        "returns_shape": list(returns.shape),
        "num_symbols": len(final_syms),
        "num_days": features.shape[0],
        "symbols": final_syms,
        "date_range": [str(dates[0]), str(dates[-1])],
        "features_path": str(out / "equity_features.npy"),
        "returns_path": str(out / "equity_returns.npy"),
    }
    logger.info("Equity pipeline complete: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_equity_pipeline()
    print(result)
