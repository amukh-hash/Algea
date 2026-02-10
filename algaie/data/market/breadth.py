"""
Market breadth indicators.

Ported from:
  - deprecated/backend_app_snapshot/data/breadth.py (calculate_ad_line, calculate_bpi)
  - deprecated/legacy_scripts/98b_build_real_breadth.py (build_breadth_daily)
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from algaie.data.common import get_close_column


# ---------------------------------------------------------------------------
# Individual indicator functions
# ---------------------------------------------------------------------------

def calculate_ad_line(close: pd.Series) -> pd.Series:
    """
    Advance / Decline line based on day-over-day sign of close price change.

    Returns a cumulative sum of +1 (up) / -1 (down) / 0 (flat).
    """
    direction = np.sign(close.diff()).fillna(0).astype(int)
    return direction.cumsum()


def calculate_bpi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Buying Pressure Index -- fraction of up-days over a rolling window.

    ``BPI in [0, 1]`` where 1 = all days up.
    """
    up = (close.diff() > 0).astype(float)
    return up.rolling(period, min_periods=1).mean()


# ---------------------------------------------------------------------------
# Cross-sectional breadth builder
# ---------------------------------------------------------------------------

def build_breadth_daily(ohlcv_frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compute daily market breadth from a dict of per-symbol OHLCV DataFrames.

    Parameters
    ----------
    ohlcv_frames : ``{symbol: DataFrame}`` -- each DataFrame must have columns
                   ``date`` and ``close`` (or ``close_adj``).

    Returns
    -------
    DataFrame with columns ``[date, market_breadth_ad, advancers, decliners, total_issues]``.
    """
    # Vectorized approach: build a single direction DataFrame, then groupby date
    direction_frames = []
    for df in ohlcv_frames.values():
        if df.empty:
            continue
        df = df.copy().sort_values("date")
        close_col = get_close_column(df)
        direction = np.sign(df[close_col].diff()).fillna(0).astype(int)
        direction_frames.append(pd.DataFrame({"date": pd.to_datetime(df["date"]), "dir": direction.values}))

    if not direction_frames:
        return pd.DataFrame(columns=["date", "market_breadth_ad", "advancers", "decliners", "total_issues"])

    all_dirs = pd.concat(direction_frames, ignore_index=True)
    grouped = all_dirs.groupby("date")["dir"]
    stats = pd.DataFrame({
        "advancers": grouped.apply(lambda s: (s > 0).sum()),
        "decliners": grouped.apply(lambda s: (s < 0).sum()),
        "total_issues": grouped.count(),
    })
    stats["market_breadth_ad"] = (stats["advancers"] - stats["decliners"]) / stats["total_issues"].clip(lower=1)
    bdf = stats.reset_index().sort_values("date")
    bdf["date"] = pd.to_datetime(bdf["date"])
    return bdf[["date", "market_breadth_ad", "advancers", "decliners", "total_issues"]]
