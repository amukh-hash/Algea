"""
VRP Data Pipeline — real historical VRP PnL and state embeddings.

Computes a Variance Risk Premium (VRP) proxy using:
    VRP = Implied Vol (VIX) − Realized Vol (SPY)

This is the structural alpha the TD3 agent learns to time.

Outputs:
    - daily_pnl.npy: simulated short-vol PnL track
    - raw_features.npy: [T, D] feature vectors for RLStateProjector
    - margin_history.npy: estimated margin utilization
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_vrp_data(period: str = "2y") -> pd.DataFrame:
    """Fetch VIX + SPY daily data from yfinance.

    Returns
    -------
    pd.DataFrame
        Aligned DataFrame with VIX, SPY close, SPY returns, etc.
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance required: pip install yfinance")

    # Fetch VIX (CBOE Volatility Index)
    vix = yf.Ticker("^VIX")
    vix_df = vix.history(period=period, interval="1d", auto_adjust=True)
    if vix_df.empty:
        raise RuntimeError("Failed to fetch VIX data")
    vix_close = vix_df[["Close"]].rename(columns={"Close": "vix_close"})

    # Fetch SPY
    spy = yf.Ticker("SPY")
    spy_df = spy.history(period=period, interval="1d", auto_adjust=True)
    if spy_df.empty:
        raise RuntimeError("Failed to fetch SPY data")
    spy_close = spy_df[["Close", "High", "Low", "Volume"]].rename(
        columns={"Close": "spy_close", "High": "spy_high", "Low": "spy_low", "Volume": "spy_volume"}
    )

    # Merge on date
    combined = pd.concat([spy_close, vix_close], axis=1, join="inner")
    combined = combined.dropna()
    logger.info("Fetched %d aligned VIX/SPY days", len(combined))
    return combined


def compute_vrp_features(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute VRP features, simulated PnL, and margin history.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: vix_close, spy_close, spy_high, spy_low, spy_volume.

    Returns
    -------
    features : np.ndarray
        Shape ``[T, D]`` feature vectors.
    pnl : np.ndarray
        Shape ``[T]`` simulated short-vol daily PnL.
    margin : np.ndarray
        Shape ``[T]`` margin utilization estimate.
    """
    spy_close = df["spy_close"]
    spy_high = df["spy_high"]
    spy_low = df["spy_low"]
    spy_volume = df["spy_volume"]
    vix = df["vix_close"]

    # SPY returns
    spy_ret = spy_close.pct_change()

    # Realized volatility (various windows, annualized)
    rv_5d = spy_ret.rolling(5).std() * np.sqrt(252)
    rv_10d = spy_ret.rolling(10).std() * np.sqrt(252)
    rv_20d = spy_ret.rolling(20).std() * np.sqrt(252)
    rv_60d = spy_ret.rolling(60).std() * np.sqrt(252)

    # VIX (implied vol) in decimal form
    iv = vix / 100.0

    # Variance Risk Premium: IV - RV
    vrp_5d = iv - rv_5d
    vrp_10d = iv - rv_10d
    vrp_20d = iv - rv_20d

    # VIX term structure proxy (VIX change)
    vix_ret = vix.pct_change()
    vix_sma5 = vix.rolling(5).mean()
    vix_sma20 = vix.rolling(20).mean()
    vix_term = (vix_sma5 - vix_sma20) / (vix_sma20 + 1e-10)

    # VIX z-score
    vix_mean20 = vix.rolling(20).mean()
    vix_std20 = vix.rolling(20).std()
    vix_zscore = (vix - vix_mean20) / (vix_std20 + 1e-10)

    # SPY momentum
    spy_mom_5d = spy_close.pct_change(5)
    spy_mom_20d = spy_close.pct_change(20)

    # Intraday range (Parkinson volatility proxy)
    intraday_range = np.log(spy_high / spy_low)
    parkinson_20d = intraday_range.rolling(20).mean() * np.sqrt(252 / (4 * np.log(2)))

    # Volume momentum
    vol_sma5 = spy_volume.rolling(5).mean()
    vol_sma20 = spy_volume.rolling(20).mean()
    vol_ratio = vol_sma5 / (vol_sma20 + 1)

    # SPY RSI
    delta = spy_close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    rsi = (100 - 100 / (1 + rs) - 50) / 50

    # Max drawdown
    rolling_max = spy_close.rolling(20).max()
    spy_drawdown = (spy_close - rolling_max) / (rolling_max + 1e-10)

    # VRP regime indicator
    vrp_zscore = vrp_20d.rolling(20).apply(
        lambda x: (x.iloc[-1] - x.mean()) / (x.std() + 1e-10), raw=False
    )

    # ── Build feature matrix ─────────────────────────────────────────
    # 64-dim feature vector built from combinations and lags
    base_features = pd.DataFrame({
        "spy_ret": spy_ret,
        "rv_5d": rv_5d,
        "rv_10d": rv_10d,
        "rv_20d": rv_20d,
        "rv_60d": rv_60d,
        "iv": iv,
        "vrp_5d": vrp_5d,
        "vrp_10d": vrp_10d,
        "vrp_20d": vrp_20d,
        "vix_ret": vix_ret,
        "vix_term": vix_term,
        "vix_zscore": vix_zscore,
        "spy_mom_5d": spy_mom_5d,
        "spy_mom_20d": spy_mom_20d,
        "parkinson_20d": parkinson_20d,
        "vol_ratio": vol_ratio,
        "rsi": rsi,
        "spy_drawdown": spy_drawdown,
        "intraday_range": intraday_range,
    })

    # Add lagged features (1d, 5d lags) to reach ~64 dims
    lag_features = {}
    for lag in [1, 5]:
        for col in ["spy_ret", "rv_5d", "iv", "vrp_5d", "vix_ret", "vix_zscore"]:
            lag_features[f"{col}_lag{lag}"] = base_features[col].shift(lag)

    # Add rolling stats
    for col in ["vrp_20d", "iv", "rv_20d"]:
        lag_features[f"{col}_min20"] = base_features[col].rolling(20).min()
        lag_features[f"{col}_max20"] = base_features[col].rolling(20).max()
        lag_features[f"{col}_skew20"] = base_features[col].rolling(20).skew()

    lag_df = pd.DataFrame(lag_features, index=base_features.index)
    all_features = pd.concat([base_features, lag_df], axis=1)

    # Drop warmup NaN
    valid = all_features.dropna()
    valid_idx = valid.index

    features_np = valid.values.astype(np.float32)
    n_feat = features_np.shape[1]
    logger.info("VRP feature dim: %d", n_feat)

    # Pad or trim to exactly 64 dims
    if n_feat < 64:
        features_np = np.pad(features_np, ((0, 0), (0, 64 - n_feat)), constant_values=0)
    elif n_feat > 64:
        features_np = features_np[:, :64]

    # ── Simulate short-vol PnL ───────────────────────────────────────
    # Simple model: short ATM put spread on SPY, collect VRP premium
    # PnL ≈ VRP_daily - |SPY_down_moves| × leverage
    vrp_daily = vrp_20d.reindex(valid_idx).values / 252  # Daily VRP premium
    spy_ret_aligned = spy_ret.reindex(valid_idx).values
    # Short vol profits when VRP > 0 and SPY doesn't crash
    pnl = (vrp_daily * 10000  # Scale: ~$100 per day of premium
           - np.clip(-spy_ret_aligned, 0, np.inf) * 50000  # Tail loss
           ).astype(np.float32)

    # ── Margin estimate ──────────────────────────────────────────────
    vix_aligned = vix.reindex(valid_idx).values
    margin = np.clip(vix_aligned / 80.0, 0.1, 1.0).astype(np.float32)

    return features_np, pnl, margin


def run_vrp_pipeline(
    period: str = "2y",
    output_dir: str = "data_lake/vrp_history",
) -> dict:
    """End-to-end VRP data pipeline.

    Returns
    -------
    dict
        Pipeline summary.
    """
    logger.info("Starting VRP data pipeline")

    # 1. Fetch
    raw = fetch_vrp_data(period=period)

    # 2. Compute features + PnL
    features, pnl, margin = compute_vrp_features(raw)

    # 3. Save
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "raw_features.npy", features)
    np.save(out / "daily_pnl.npy", pnl)
    np.save(out / "margin_history.npy", margin)

    summary = {
        "status": "ok",
        "features_shape": list(features.shape),
        "pnl_shape": list(pnl.shape),
        "features_path": str(out / "raw_features.npy"),
        "pnl_path": str(out / "daily_pnl.npy"),
        "margin_path": str(out / "margin_history.npy"),
        "pnl_total": float(pnl.sum()),
        "pnl_sharpe": float(pnl.mean() / (pnl.std() + 1e-10) * np.sqrt(252)),
    }
    logger.info("VRP pipeline complete: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = run_vrp_pipeline()
    print(result)
