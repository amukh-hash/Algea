"""
Series builder for Lag-Llama risk forecaster.

Derives target series from daily close prices:
  - sqret: clipped squared log returns (primary)
  - abs_neg_ret: absolute negative returns
  - EWMA variance baseline for fallback / blending
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def build_log_returns(close: pd.Series) -> pd.Series:
    """Compute daily log returns from close prices."""
    lr = np.log(close / close.shift(1))
    return lr.dropna()


def build_sqret_series(
    close: pd.Series,
    upper_clip_percentile: float = 99.5,
) -> pd.Series:
    """Build squared return series (primary forecast target).

    sqret_t = clip(r_t^2, upper_percentile)
    """
    lr = build_log_returns(close)
    sqret = lr ** 2
    clip_val = float(np.nanpercentile(sqret.dropna().values, upper_clip_percentile))
    return sqret.clip(upper=clip_val).rename("sqret")


def build_abs_neg_ret_series(close: pd.Series) -> pd.Series:
    """Build absolute negative return series (secondary target).

    abs_neg_ret_t = |min(r_t, 0)|
    """
    lr = build_log_returns(close)
    return lr.clip(upper=0.0).abs().rename("abs_neg_ret")


def build_ewma_variance(
    close: pd.Series,
    span: int = 20,
) -> pd.Series:
    """EWMA variance estimate (deterministic baseline for fallback).

    Returns annualised EWMA variance: ewma_var * 252
    """
    lr = build_log_returns(close)
    sqret = lr ** 2
    ewma_var = sqret.ewm(span=span, min_periods=max(span // 2, 5)).mean()
    return (ewma_var * 252).rename("ewma_var_ann")


def ewma_rv_quantiles(
    ewma_var_ann: pd.Series,
    quantiles: tuple = (0.50, 0.90, 0.95, 0.99),
    lookback: int = 252,
) -> dict[float, float]:
    """Compute empirical quantiles of ewma_rv from trailing distribution.

    Used as deterministic baseline for comparison / blending.
    """
    rv = np.sqrt(ewma_var_ann)
    tail = rv.dropna().iloc[-lookback:]
    if len(tail) < 20:
        # Not enough data — return current value for all quantiles
        curr = float(rv.iloc[-1]) if len(rv) > 0 else 0.15
        return {q: curr for q in quantiles}
    return {q: float(np.nanpercentile(tail.values, q * 100)) for q in quantiles}


def save_series(
    series: pd.Series,
    root: Path,
    as_of_date: str,
    underlying: str,
    series_type: str,
) -> Path:
    """Persist derived series as Parquet."""
    out_dir = root / "lag_llama" / "series" / f"date={as_of_date}" / f"underlying={underlying}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{series_type}.parquet"
    df = series.to_frame()
    df.index.name = "date"
    df.to_parquet(path)
    return path


def load_series(
    root: Path,
    as_of_date: str,
    underlying: str,
    series_type: str,
) -> Optional[pd.Series]:
    """Load previously persisted series."""
    path = root / "lag_llama" / "series" / f"date={as_of_date}" / f"underlying={underlying}" / f"{series_type}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    return df.iloc[:, 0]
