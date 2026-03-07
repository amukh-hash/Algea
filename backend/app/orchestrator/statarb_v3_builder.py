"""
StatArb V3 Live Feature Builder (Staged for Post-Freeze Deployment)

Constructs the ``[1, 60, 10]`` tensor for the iTransformer using
10 Idiosyncratic Pairs + Order Flow Imbalance (OFI).

Pair order MUST match ``statarb_v3_pipeline.py`` training exactly:
  0: KRE/IWM   (Regional Banks vs Small Caps)
  1: XBI/IWM   (Biotech vs Small Caps)
  2: ARKK/QQQ  (Hyper-growth vs Broad Tech)
  3: SMH/QQQ   (Semiconductors vs Broad Tech)
  4: GDXJ/GLD  (Junior Miners vs Physical Gold)
  5: XOP/USO   (Oil Explorers vs Physical Oil)
  6: ITB/VNQ   (Homebuilders vs REITs)
  7: JNK/TLT   (High Yield vs Treasuries)
  8: TAN/XLE   (Solar vs Traditional Energy)
  9: XRT/SPY   (Retail vs Broad Market)

On failure, returns a zero tensor to neutralize StatArb signal generation.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)

PAIRS_V3 = [
    ("KRE", "IWM"),
    ("XBI", "IWM"),
    ("ARKK", "QQQ"),
    ("SMH", "QQQ"),
    ("GDXJ", "GLD"),
    ("XOP", "USO"),
    ("ITB", "VNQ"),
    ("JNK", "TLT"),
    ("TAN", "XLE"),
    ("XRT", "SPY"),
]

_N_PAIRS = len(PAIRS_V3)
_LOOKBACK = 60  # 60 bars × 60 min = 8.5 trading days
_LOOKBACK_Z = 60  # Rolling Z-score window


def build_live_statarb_v3_state(
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Construct the ``[1, 60, 10]`` live tensor for StatArb V3.

    Requires ~20 days of 60-min bar history to buffer rolling Z-scores
    and OFI normalization.

    Returns
    -------
    torch.Tensor
        Shape ``[1, 60, 10]`` in ``bfloat16`` on *device*.
    """
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance required for StatArb V3 live features.")
        return _zero_tensor(device)

    tickers = sorted(set(sym for pair in PAIRS_V3 for sym in pair))

    try:
        df = yf.download(tickers, period="20d", interval="60m", progress=False)
        closes = df["Close"].ffill().dropna()
        volumes = df["Volume"].ffill().dropna()

        if len(closes) < _LOOKBACK + 10:
            logger.error(
                "Insufficient 60m bars: got %d, need %d+",
                len(closes), _LOOKBACK + 10,
            )
            return _zero_tensor(device)

        # ── OFI (Order Flow Imbalance) Z-Scores ──
        returns = closes.pct_change()
        ofi = returns.apply(np.sign) * volumes
        ofi_z = (ofi - ofi.rolling(10).mean()) / (ofi.rolling(10).std() + 1e-8)

        # ── Combined Signal for all 10 pairs ──
        z_scores_df = pd.DataFrame(index=closes.index)

        for sym_a, sym_b in PAIRS_V3:
            ratio = np.log(closes[sym_a]) - np.log(closes[sym_b])
            r_mean = ratio.rolling(window=_LOOKBACK_Z).mean()
            r_std = ratio.rolling(window=_LOOKBACK_Z).std()
            spread_z = (ratio - r_mean) / (r_std + 1e-8)

            combined_signal = spread_z + (0.25 * ofi_z[sym_a]) - (0.25 * ofi_z[sym_b])
            z_scores_df[f"{sym_a}_{sym_b}"] = combined_signal

        z_scores_df = z_scores_df.dropna()

        if len(z_scores_df) < _LOOKBACK:
            logger.error(
                "Not enough clean bars: got %d, need %d",
                len(z_scores_df), _LOOKBACK,
            )
            return _zero_tensor(device)

        # Last 60 bars → [60, 10]
        latest = z_scores_df.iloc[-_LOOKBACK:].values.astype(np.float32)

        logger.info(
            "StatArb V3 state: %d bars × %d pairs  "
            "spread_range=[%.2f, %.2f]",
            latest.shape[0], latest.shape[1],
            latest.min(), latest.max(),
        )

        t = torch.tensor(latest, dtype=torch.float32).unsqueeze(0)
        if device is not None:
            t = t.to(device)
        return t

    except Exception as e:
        logger.exception("Failed to build StatArb V3 state: %s. Using zero tensor fallback.", e)
        return _zero_tensor(device)


def _zero_tensor(device: Optional[torch.device]) -> torch.Tensor:
    """Neutral zero tensor — disables StatArb signal generation."""
    t = torch.zeros(1, _LOOKBACK, _N_PAIRS, dtype=torch.float32)
    if device is not None:
        t = t.to(device)
    return t
