"""
Market covariates builder -- SPY/QQQ/IWM returns, VIX proxy, rate proxy.

Ported from deprecated/legacy_scripts/98_build_real_covariates.py.
Decoupled from ``pathmap`` -- accepts DataFrames directly.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from algea.data.common import get_close_column

logger = logging.getLogger(__name__)

# Canonical covariates schema
CANONICAL_COV_COLS = [
    "date", "spy_ret_1d", "qqq_ret_1d", "iwm_ret_1d",
    "rv21_level", "rv21_chg_1d", "ief_ret_1d",
]


def _prepare_symbol_returns(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Compute 1-day return for a single symbol from OHLCV data."""
    if df.empty:
        return pd.DataFrame()
    df = df.copy().sort_values("date")
    df = df.set_index("date")
    df = df[~df.index.duplicated(keep="first")]
    close_col = get_close_column(df)
    ret = df[close_col].pct_change().rename(f"{name}_ret_1d")
    return ret.to_frame()


def _load_symbol(per_ticker_dir: Path, symbol: str) -> pd.DataFrame:
    """Load a per-ticker parquet file, returning empty DataFrame if missing."""
    fp = per_ticker_dir / f"{symbol}.parquet"
    if not fp.exists():
        logger.warning(f"Per-ticker file not found: {fp}")
        return pd.DataFrame()
    return pd.read_parquet(fp)


def _build_base(
    spy_df: pd.DataFrame,
    qqq_df: pd.DataFrame,
    iwm_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Shared logic: join SPY/QQQ/IWM returns into a single frame."""
    spy_f = _prepare_symbol_returns(spy_df, "spy")
    qqq_f = _prepare_symbol_returns(qqq_df, "qqq")

    df = spy_f.copy()
    df = df.join(qqq_f, how="outer")

    if iwm_df is not None and not iwm_df.empty:
        iwm_f = _prepare_symbol_returns(iwm_df, "iwm")
        df = df.join(iwm_f, how="outer")
    else:
        df["iwm_ret_1d"] = 0.0

    return df


def build_covariates(
    spy_df: pd.DataFrame,
    qqq_df: pd.DataFrame,
    iwm_df: pd.DataFrame | None = None,
    ief_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Build market covariates DataFrame (legacy schema).

    Returns
    -------
    DataFrame with columns:
      ``[date, spy_ret_1d, qqq_ret_1d, iwm_ret_1d, vix_level, rate_proxy]``
    """
    df = _build_base(spy_df, qqq_df, iwm_df)

    # SPY-based VIX proxy: 21-day annualised realised vol x 100
    vix_s = (df["spy_ret_1d"].rolling(window=21).std() * np.sqrt(252) * 100).rename("vix_level")
    df = df.join(vix_s, how="left")

    # Rate proxy -- IEF close price (inverse yield)
    if ief_df is not None and not ief_df.empty:
        ief_tmp = ief_df.copy().sort_values("date").set_index("date")
        ief_tmp = ief_tmp[~ief_tmp.index.duplicated(keep="first")]
        close_col = get_close_column(ief_tmp)
        rate_s = ief_tmp[close_col].reindex(df.index).rename("rate_proxy")
        df = df.join(rate_s, how="left")
    else:
        df["rate_proxy"] = np.nan

    df = df.sort_index().reset_index()
    df = df.rename(columns={"index": "date"})

    # Forward-fill levels
    df["vix_level"] = df["vix_level"].ffill()
    df["rate_proxy"] = df["rate_proxy"].ffill()

    # Returns fill NaN -> 0
    for c in ["spy_ret_1d", "qqq_ret_1d", "iwm_ret_1d"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    return df


# --- Canonical covariates (v2) ---

def build_canonical_covariates(
    spy_df: pd.DataFrame,
    qqq_df: pd.DataFrame,
    iwm_df: Optional[pd.DataFrame] = None,
    ief_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build canonical market covariates (stationary schema for Chronos-2 training).

    Schema
    ------
    date, spy_ret_1d, qqq_ret_1d, iwm_ret_1d, rv21_level, rv21_chg_1d, ief_ret_1d
    """
    df = _build_base(spy_df, qqq_df, iwm_df)

    # RV21: 21-day annualised realised vol of SPY x 100
    rv21 = (
        df["spy_ret_1d"].rolling(window=21, min_periods=21).std()
        * np.sqrt(252) * 100
    ).rename("rv21_level")
    df = df.join(rv21, how="left")

    # rv21_chg_1d: pct_change of rv21_level (stationary)
    rv21_chg = df["rv21_level"].pct_change().rename("rv21_chg_1d")
    rv21_chg = rv21_chg.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    df["rv21_chg_1d"] = rv21_chg

    # IEF return (stationary) instead of raw price level
    if ief_df is not None and not ief_df.empty:
        ief_f = _prepare_symbol_returns(ief_df, "ief")
        df = df.join(ief_f, how="left")
    else:
        df["ief_ret_1d"] = 0.0

    df = df.sort_index().reset_index()
    df = df.rename(columns={"index": "date"})

    # Forward-fill levels: initialize with first non-NaN then ffill
    first_valid = df["rv21_level"].first_valid_index()
    if first_valid is not None:
        first_val = df.loc[first_valid, "rv21_level"]
        df.loc[:first_valid, "rv21_level"] = first_val
    df["rv21_level"] = df["rv21_level"].ffill()

    # Return columns: NaN -> 0
    for c in ["spy_ret_1d", "qqq_ret_1d", "iwm_ret_1d", "rv21_chg_1d", "ief_ret_1d"]:
        if c in df.columns:
            df[c] = df[c].fillna(0.0)

    # Keep only canonical columns
    df = df[CANONICAL_COV_COLS].copy()
    return df


def build_and_persist_covariates(
    per_ticker_dir: Path,
    out_path: Path,
    overwrite: bool = False,
) -> pd.DataFrame:
    """
    Build canonical covariates parquet from per-ticker OHLCV files.

    Reads SPY, QQQ, IWM, IEF from ``per_ticker_dir/{SYMBOL}.parquet``
    and writes to ``out_path``.
    """
    if out_path.exists() and not overwrite:
        logger.info(f"Covariates already exist at {out_path}, loading.")
        return pd.read_parquet(out_path)

    spy_df = _load_symbol(per_ticker_dir, "SPY")
    qqq_df = _load_symbol(per_ticker_dir, "QQQ")
    iwm_df = _load_symbol(per_ticker_dir, "IWM")
    ief_df = _load_symbol(per_ticker_dir, "IEF")

    if spy_df.empty or qqq_df.empty:
        raise FileNotFoundError(
            f"SPY and QQQ per-ticker parquets required in {per_ticker_dir}"
        )

    cov = build_canonical_covariates(spy_df, qqq_df, iwm_df, ief_df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cov.to_parquet(out_path, index=False)
    logger.info(
        f"Canonical covariates: {len(cov)} rows, "
        f"{cov['date'].min()} -> {cov['date'].max()} -> {out_path}"
    )
    return cov
