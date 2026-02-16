"""Canonical session bars: provider-agnostic open/close extraction.

CHANGE LOG (2026-02-14):
  - NEW MODULE: D1 of CO→OC integrity refactor.
  - SessionSpec dataclass for session definition.
  - build_session_bars(): derives session open/close prices from raw bars.
  - validate_session_bars(): uniqueness, ordering, finite-price assertions.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, time, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session specification
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SessionSpec:
    """Define a trading session's timezone and open/close boundaries.

    Parameters
    ----------
    timezone : str
        IANA timezone name, e.g. ``"America/New_York"``.
    open_time : str
        Session open time as ``"HH:MM"`` in *timezone*.
    close_time : str
        Session close time as ``"HH:MM"`` in *timezone*.
    open_tolerance_mins : int
        Tolerance window (minutes) after open_time to search for first price.
    close_tolerance_mins : int
        Tolerance window (minutes) before close_time to search for last price.
    safety_margin_secs : int
        Feature cutoff safety margin: features must have timestamps strictly
        before ``session_open_ts - safety_margin_secs`` to avoid leakage.
    """

    timezone: str = "America/New_York"
    open_time: str = "09:30"
    close_time: str = "16:00"
    open_tolerance_mins: int = 5
    close_tolerance_mins: int = 5
    safety_margin_secs: int = 0

    @property
    def tz(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)

    @property
    def open_time_parsed(self) -> time:
        parts = self.open_time.split(":")
        return time(int(parts[0]), int(parts[1]))

    @property
    def close_time_parsed(self) -> time:
        parts = self.close_time.split(":")
        return time(int(parts[0]), int(parts[1]))


# ---------------------------------------------------------------------------
# Default US Equity Futures session
# ---------------------------------------------------------------------------

US_EQUITY_SESSION = SessionSpec(
    timezone="America/New_York",
    open_time="09:30",
    close_time="16:00",
    open_tolerance_mins=5,
    close_tolerance_mins=5,
    safety_margin_secs=60,
)


# ---------------------------------------------------------------------------
# Build session bars
# ---------------------------------------------------------------------------

def build_session_bars(
    raw: pd.DataFrame,
    spec: SessionSpec,
    *,
    root_col: str = "root",
    ts_col: str = "timestamp",
    price_col: str = "price",
    volume_col: Optional[str] = None,
) -> pd.DataFrame:
    """Derive session open/close prices from raw intraday or daily bars.

    For **daily bars** (one row per root per day with open/high/low/close),
    the function simply renames open → session_open_price, close →
    session_close_price and synthesizes session timestamps from the spec.

    For **intraday bars** (multiple rows per day), it finds the first price
    at/after session open and the last price at/before session close.

    Parameters
    ----------
    raw
        Input DataFrame.  Must contain *root_col* and *ts_col*.
        For daily bars: must also have ``open``, ``close`` columns.
        For intraday: must have *price_col*.
    spec
        Session specification.
    root_col, ts_col, price_col
        Column name overrides.
    volume_col
        If provided, aggregate session volume.

    Returns
    -------
    DataFrame with columns:
        root, trading_day, session_open_price, session_close_price,
        session_open_ts, session_close_ts, open_missing, close_missing
        (plus volume if volume_col was provided)

    All timestamps are tz-naive in *spec.timezone* convention.
    """
    df = raw.copy()
    tz = spec.tz

    # Detect daily vs intraday
    is_daily = "open" in df.columns and "close" in df.columns

    if is_daily:
        return _from_daily_bars(df, spec, root_col=root_col, ts_col=ts_col,
                                volume_col=volume_col)
    else:
        return _from_intraday_bars(df, spec, root_col=root_col, ts_col=ts_col,
                                   price_col=price_col, volume_col=volume_col)


def _from_daily_bars(
    df: pd.DataFrame,
    spec: SessionSpec,
    *,
    root_col: str,
    ts_col: str,
    volume_col: Optional[str],
) -> pd.DataFrame:
    """Convert daily OHLCV bars to session bars."""
    tz = spec.tz
    open_t = spec.open_time_parsed
    close_t = spec.close_time_parsed

    out = df[[root_col]].copy()
    out = out.rename(columns={root_col: "root"})

    # Derive trading_day
    if "trading_day" in df.columns:
        out["trading_day"] = pd.to_datetime(df["trading_day"]).dt.date
    elif ts_col in df.columns:
        ts = pd.to_datetime(df[ts_col])
        if ts.dt.tz is not None:
            ts = ts.dt.tz_convert(tz).dt.tz_localize(None)
        out["trading_day"] = ts.dt.date
    else:
        raise ValueError(f"Cannot derive trading_day: no '{ts_col}' or 'trading_day' column")

    out["session_open_price"] = df["open"].values
    out["session_close_price"] = df["close"].values

    # Synthesize session timestamps (tz-naive in spec timezone)
    out["session_open_ts"] = out["trading_day"].apply(
        lambda d: pd.Timestamp.combine(d, open_t)
    )
    out["session_close_ts"] = out["trading_day"].apply(
        lambda d: pd.Timestamp.combine(d, close_t)
    )

    out["open_missing"] = df["open"].isna().values
    out["close_missing"] = df["close"].isna().values

    if volume_col and volume_col in df.columns:
        out["volume"] = df[volume_col].values
    elif "volume" in df.columns:
        out["volume"] = df["volume"].values

    # Stable sort
    out = out.sort_values(["root", "trading_day"]).reset_index(drop=True)
    return out


def _from_intraday_bars(
    df: pd.DataFrame,
    spec: SessionSpec,
    *,
    root_col: str,
    ts_col: str,
    price_col: str,
    volume_col: Optional[str],
) -> pd.DataFrame:
    """Extract session open/close from intraday tick/bar data."""
    tz = spec.tz
    open_t = spec.open_time_parsed
    close_t = spec.close_time_parsed
    open_tol = timedelta(minutes=spec.open_tolerance_mins)
    close_tol = timedelta(minutes=spec.close_tolerance_mins)

    ts = pd.to_datetime(df[ts_col])
    if ts.dt.tz is not None:
        ts_local = ts.dt.tz_convert(tz).dt.tz_localize(None)
    else:
        ts_local = ts

    work = df.copy()
    work["_ts_local"] = ts_local
    work["_trading_day"] = ts_local.dt.date
    work["_time"] = ts_local.dt.time

    rows = []
    for (root, day), grp in work.groupby([root_col, "_trading_day"], sort=True):
        grp = grp.sort_values("_ts_local")

        # Session open window
        open_start = pd.Timestamp.combine(day, open_t)
        open_end = open_start + open_tol
        open_mask = (grp["_ts_local"] >= open_start) & (grp["_ts_local"] <= open_end)
        open_rows = grp[open_mask]

        # Session close window
        close_end = pd.Timestamp.combine(day, close_t)
        close_start = close_end - close_tol
        close_mask = (grp["_ts_local"] >= close_start) & (grp["_ts_local"] <= close_end)
        close_rows = grp[close_mask]

        row = {
            "root": root,
            "trading_day": day,
            "session_open_price": open_rows[price_col].iloc[0] if len(open_rows) > 0 else np.nan,
            "session_close_price": close_rows[price_col].iloc[-1] if len(close_rows) > 0 else np.nan,
            "session_open_ts": open_rows["_ts_local"].iloc[0] if len(open_rows) > 0 else pd.NaT,
            "session_close_ts": close_rows["_ts_local"].iloc[-1] if len(close_rows) > 0 else pd.NaT,
            "open_missing": len(open_rows) == 0,
            "close_missing": len(close_rows) == 0,
        }

        if volume_col and volume_col in grp.columns:
            session_mask = (grp["_ts_local"] >= open_start) & (grp["_ts_local"] <= close_end)
            row["volume"] = grp.loc[session_mask, volume_col].sum()

        rows.append(row)

    result = pd.DataFrame(rows)
    result = result.sort_values(["root", "trading_day"]).reset_index(drop=True)
    return result


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_session_bars(
    df: pd.DataFrame,
    *,
    root_col: str = "root",
) -> None:
    """Assert session bar integrity.

    Checks:
    1. Uniqueness on (root, trading_day).
    2. session_open_ts < session_close_ts where both present.
    3. Prices are finite where present.

    Raises
    ------
    ValueError
        With descriptive message if any check fails.
    """
    # 1. Uniqueness
    dups = df.duplicated(subset=[root_col, "trading_day"], keep=False)
    if dups.any():
        examples = df[dups][[root_col, "trading_day"]].head(5).to_string()
        raise ValueError(
            f"Duplicate (root, trading_day) rows found in session bars:\n{examples}"
        )

    # 2. Timestamp ordering
    has_both = df["session_open_ts"].notna() & df["session_close_ts"].notna()
    if has_both.any():
        bad_order = df[has_both & (df["session_open_ts"] >= df["session_close_ts"])]
        if len(bad_order) > 0:
            examples = bad_order[[root_col, "trading_day",
                                  "session_open_ts", "session_close_ts"]].head(5).to_string()
            raise ValueError(
                f"session_open_ts >= session_close_ts in {len(bad_order)} rows:\n{examples}"
            )

    # 3. Finite prices
    for col in ("session_open_price", "session_close_price"):
        if col in df.columns:
            present = df[col].notna()
            if present.any():
                non_finite = present & ~np.isfinite(df[col].astype(float))
                if non_finite.any():
                    examples = df[non_finite][[root_col, "trading_day", col]].head(5).to_string()
                    raise ValueError(
                        f"Non-finite values in {col} for {non_finite.sum()} rows:\n{examples}"
                    )

    logger.info(
        "Session bars validated: %d rows, %d roots, %d days",
        len(df),
        df[root_col].nunique(),
        df["trading_day"].nunique(),
    )
