"""
Volatility scaling for the portfolio return stream.

Uses rolling realized vol to compute leverage that targets a user-specified
annualized vol.  Long-only default caps leverage at 1.0 (no leverage).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VolTargetConfig:
    """Volatility-targeting parameters."""

    target_vol_ann: float = 0.15   # annualized target vol
    lookback_periods: int = 12     # rolling window (12 × 10d ≈ 120 trading days)
    max_leverage: float = 1.0      # default: no leverage for long-only
    min_leverage: float = 0.0      # can go to cash


def compute_leverage(
    returns: np.ndarray,
    idx: int,
    cfg: VolTargetConfig,
    periods_per_year: float = 25.2,
) -> float:
    """Compute leverage for period *idx* based on trailing realized vol.

    Parameters
    ----------
    returns : array
        Full return history up to (but not including) period *idx*.
    idx : int
        Current period index.
    cfg : VolTargetConfig
        Vol-targeting parameters.
    periods_per_year : float
        Annualization factor (e.g. 252/10 = 25.2).

    Returns
    -------
    lev : float
        Leverage factor, clipped to [min_leverage, max_leverage].
    """
    if idx < cfg.lookback_periods:
        # Not enough history → default leverage = 1.0 (or min if > 1)
        return min(1.0, cfg.max_leverage)

    window = returns[idx - cfg.lookback_periods : idx]
    realized_vol = float(np.std(window, ddof=1)) * np.sqrt(periods_per_year)

    if realized_vol < 1e-8:
        lev = cfg.max_leverage
    else:
        lev = cfg.target_vol_ann / realized_vol

    lev = float(np.clip(lev, cfg.min_leverage, cfg.max_leverage))
    return lev


def apply_leverage(gross_return: float, lev: float) -> float:
    """Scale a gross return by a leverage factor.

    Parameters
    ----------
    gross_return : float
        Unscaled portfolio return for the period.
    lev : float
        Leverage multiplier.

    Returns
    -------
    scaled_return : float
    """
    return gross_return * lev
