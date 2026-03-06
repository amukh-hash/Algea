"""
Build the 10-dimensional live macro feature vector for the TD3 Risk Agent.

Matches the ``train_rl_macro.py`` training features exactly:
  [SPY_ret_1d, SPY_ret_5d, SPY_vol_21d, VIX_change, VIX_level,
   TNX_shock, USD_shock, Credit_Spread, Trend_200d, Drift_Proxy]

If data fetching fails or NaNs are detected, returns a synthetic
"Nuclear Crash" state that forces the agent to veto and paralyze margin.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

# Synthetic "Nuclear Crash" state — high vol, rising VIX, yield shock,
# below 200d SMA.  Forces the TD3 Actor into maximum veto.
_NUCLEAR_CRASH_STATE = np.array(
    [-0.05, -0.10, 0.40, 0.50, 0.40, 0.1, 0.05, -0.05, -0.10, 0.40],
    dtype=np.float32,
)


def build_live_rl_state(
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Construct the 10-dim live macro feature vector.

    Requires ~2 years of historical daily closes for the 200-day SMA
    and 60-day rolling drift proxy.

    Returns
    -------
    torch.Tensor
        Shape ``[1, 10]`` on *device*.
    """
    try:
        import pandas as pd
        import yfinance as yf
    except ImportError as e:
        logger.error("Missing dependency for RL state: %s. Using crash state.", e)
        return _tensor(_NUCLEAR_CRASH_STATE, device)

    tickers = ["SPY", "^VIX", "^TNX", "UUP", "HYG", "LQD"]

    try:
        df = yf.download(tickers, period="2y", interval="1d", progress=False)["Close"]
        df = df.ffill().dropna()

        if len(df) < 210:
            logger.error("Insufficient data (%d rows). Using crash state.", len(df))
            return _tensor(_NUCLEAR_CRASH_STATE, device)

        # ── Base Features ──
        spy_ret_1d = df["SPY"].pct_change()
        spy_ret_5d = df["SPY"].pct_change(5)
        spy_vol_21d = spy_ret_1d.rolling(21).std() * np.sqrt(252)
        vix_change = df["^VIX"].pct_change()
        vix_level = df["^VIX"] / 100.0

        # ── Macro Features ──
        tnx_shock = df["^TNX"].diff(5) / 10.0
        usd_shock = df["UUP"].pct_change(5)
        credit_spread = (df["HYG"] / df["LQD"]).pct_change(21)

        # ── Regime Features ──
        trend_200d = (df["SPY"] / df["SPY"].rolling(200).mean()) - 1.0
        drift_proxy = spy_vol_21d.rolling(60).mean()

        # Aggregate latest row
        latest = pd.DataFrame({
            "SPY_ret_1d": spy_ret_1d,
            "SPY_ret_5d": spy_ret_5d,
            "SPY_vol_21d": spy_vol_21d,
            "VIX_change": vix_change,
            "VIX_level": vix_level,
            "TNX_shock": tnx_shock,
            "USD_shock": usd_shock,
            "Credit_Spread": credit_spread,
            "Trend_200d": trend_200d,
            "Drift_Proxy": drift_proxy,
        }).iloc[-1].values.astype(np.float32)

        if np.isnan(latest).any() or np.isinf(latest).any():
            logger.error("NaN/Inf in live RL macro features. Using crash state.")
            return _tensor(_NUCLEAR_CRASH_STATE, device)

        logger.info(
            "RL state built: SPY_1d=%.4f VIX=%.1f TNX_Δ=%.3f Credit=%.4f",
            latest[0], latest[4] * 100, latest[5], latest[7],
        )
        return _tensor(latest, device)

    except Exception as e:
        logger.error("Failed to build RL state: %s. Using crash state.", e)
        return _tensor(_NUCLEAR_CRASH_STATE, device)


def _tensor(arr: np.ndarray, device: Optional[torch.device]) -> torch.Tensor:
    t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
    if device is not None:
        t = t.to(device)
    return t
