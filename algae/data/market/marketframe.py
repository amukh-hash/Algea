"""
MarketFrame builder -- joins per-symbol OHLCV with covariates and breadth.

Ported from deprecated/backend_app_snapshot/data/marketframe.py.
Decoupled from ``pathmap`` / ``calendar`` -- accepts DataFrames directly.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd

from algae.data.common import ensure_datetime


def build_marketframe(
    ohlcv_frames: Dict[str, pd.DataFrame],
    covariates: pd.DataFrame,
    breadth: pd.DataFrame,
    trading_days: Optional[List] = None,
) -> pd.DataFrame:
    """
    Construct a panel DataFrame aligned to trading days.

    Parameters
    ----------
    ohlcv_frames : ``{symbol: DataFrame}`` with columns ``[date, open, high, low, close, volume]``
    covariates : output of ``build_covariates()``
    breadth : output of ``build_breadth_daily()``
    trading_days : optional list of dates to reindex to (NYSE calendar)

    Returns
    -------
    Long-format DataFrame with ``[date, symbol, OHLCV cols, covariate cols, breadth cols]``
    """
    cov = ensure_datetime(covariates.copy())
    brd = ensure_datetime(breadth.copy())

    frames: List[pd.DataFrame] = []
    for symbol, ohlcv in ohlcv_frames.items():
        if ohlcv.empty:
            continue
        df = ensure_datetime(ohlcv.copy())
        df = df.sort_values("date")

        if trading_days is not None:
            idx = pd.DatetimeIndex(trading_days)
            df = df.set_index("date").reindex(idx).reset_index().rename(columns={"index": "date"})

        df["symbol"] = symbol
        merged = df.merge(cov, on="date", how="left")
        merged = merged.merge(brd, on="date", how="left")
        frames.append(merged)

    if not frames:
        raise ValueError("MarketFrame build produced no data.")

    return pd.concat(frames, ignore_index=True)
