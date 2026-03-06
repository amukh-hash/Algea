"""
Research Cycle 3: StatArb V3 (Idiosyncratic Pairs + OFI)
Timeframe: 60-minute bars (730-day history). Target: cuda:0.

Pivots away from macro SPDR ETFs (proven efficient at 50.3% DA) to
10 idiosyncratic/thematic pairs where structural liquidity inefficiencies
exist (e.g., Regional Banks vs Small Caps during banking panics).

Injects Order Flow Imbalance (OFI) proxy: sign(return) × volume,
normalized as a rolling Z-score, blended into the cointegration spread.
The iTransformer now sees Price Divergence + Volume Exhaustion simultaneously.
"""
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StatArb_V3")

# 10 Idiosyncratic / Thematic Pairs (Target vs Hedge)
PAIRS = [
    ("KRE", "IWM"),   # Regional Banks vs Small Caps (Distress vs Broad)
    ("XBI", "IWM"),   # Biotech vs Small Caps (Retail Flow vs Broad)
    ("ARKK", "QQQ"),  # Hyper-growth vs Broad Tech (Duration Risk)
    ("SMH", "QQQ"),   # Semiconductors vs Broad Tech (Momentum)
    ("GDXJ", "GLD"),  # Junior Miners vs Physical Gold (Miners Leverage)
    ("XOP", "USO"),   # Oil Explorers vs Physical Oil (Equity vs Commodity)
    ("ITB", "VNQ"),   # Homebuilders vs REITs (Rate Sensitivity)
    ("JNK", "TLT"),   # High Yield vs Treasuries (Credit Stress)
    ("TAN", "XLE"),   # Solar vs Traditional Energy (Policy/Thematic)
    ("XRT", "SPY"),   # Retail vs Broad Market (Consumer Health)
]


def build_statarb_v3_dataset():
    logger.info("Fetching 60-minute data for 20 unique tickers (730-day limit)...")
    tickers = sorted(set(sym for pair in PAIRS for sym in pair))
    logger.info("  Tickers: %s", tickers)

    df = yf.download(tickers, period="730d", interval="60m", progress=False)
    closes = df["Close"].ffill().dropna()
    volumes = df["Volume"].ffill().dropna()
    logger.info("  Raw: %d bars × %d tickers", len(closes), closes.shape[1])

    # ── OFI (Order Flow Imbalance) Proxy ──
    # Estimated buying/selling pressure: sign(Return) × Volume
    returns = closes.pct_change()
    ofi = returns.apply(np.sign) * volumes
    # Rolling 10-bar OFI Z-score to normalize volume spikes
    ofi_z = (ofi - ofi.rolling(10).mean()) / (ofi.rolling(10).std() + 1e-8)

    variates = len(PAIRS)
    z_scores_df = pd.DataFrame(index=closes.index)
    lookback_z = 60  # 60 hours ≈ 8.5 trading days

    logger.info("Calculating Cointegration Z-Scores and Injecting OFI...")
    for sym_a, sym_b in PAIRS:
        # Log price ratio (the spread)
        ratio = np.log(closes[sym_a]) - np.log(closes[sym_b])

        # Rolling Z-score of the spread
        r_mean = ratio.rolling(window=lookback_z).mean()
        r_std = ratio.rolling(window=lookback_z).std()
        spread_z = (ratio - r_mean) / (r_std + 1e-8)

        # Inject Order Flow Imbalance:
        # If spread is overextended AND OFI on Sym A is heavily negative,
        # it strengthens the mean-reversion signal.
        combined_signal = spread_z + (0.25 * ofi_z[sym_a]) - (0.25 * ofi_z[sym_b])

        z_scores_df[f"{sym_a}_{sym_b}"] = combined_signal

    z_scores_df = z_scores_df.dropna()

    # ── Target: Forward 12-bar Δ Z-score ──
    # 12 bars at 60-min = 1.7 trading days
    targets_df = z_scores_df.shift(-12) - z_scores_df

    valid_idx = targets_df.dropna().index
    z_scores_df = z_scores_df.loc[valid_idx]
    targets_df = targets_df.loc[valid_idx]

    N = len(z_scores_df)
    logger.info("  Valid samples: %d", N)

    # ── Build 3D Tensor [Samples, Lookback=60, Variates=10] ──
    lookback_tsfm = 60
    n_samples = N - lookback_tsfm
    X = np.zeros((n_samples, lookback_tsfm, variates), dtype=np.float32)
    y = np.zeros((n_samples, variates), dtype=np.float32)

    z_values = z_scores_df.values
    t_values = targets_df.values

    for i in range(lookback_tsfm, N):
        X[i - lookback_tsfm] = z_values[i - lookback_tsfm:i]
        y[i - lookback_tsfm] = t_values[i]

    # Clip extreme targets
    y = np.clip(y, -5.0, 5.0)

    out_dir = Path("data_lake/statarb_v3")
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "X_features.npy", X)
    np.save(out_dir / "y_targets.npy", y)

    # Save pairs manifest
    pd.Series([f"{a}_{b}" for a, b in PAIRS]).to_csv(
        out_dir / "pairs_manifest.csv", index=False,
    )

    logger.info("StatArb V3 Data Complete: X=%s  y=%s", X.shape, y.shape)
    logger.info("  y stats: mean=%.4f std=%.4f range=[%.2f, %.2f]",
                y.mean(), y.std(), y.min(), y.max())


if __name__ == "__main__":
    build_statarb_v3_dataset()
