"""Time-aware downsampling helpers for TSFM/Chronos pipelines."""

from __future__ import annotations

import re
from datetime import timedelta

import pandas as pd

_DURATION_RE = re.compile(r"^(?P<n>\d+)(?P<u>min|h|s)$")


def parse_duration_to_timedelta(dur_str: str) -> timedelta:
    """Parse duration strings like ``5min``, ``1h``, ``30s``.

    Raises:
        ValueError: if the input format is invalid.
    """

    match = _DURATION_RE.match(str(dur_str).strip())
    if not match:
        raise ValueError(
            f"invalid duration '{dur_str}': expected <int>min, <int>h, or <int>s (e.g. '5min', '1h', '30s')"
        )
    n = int(match.group("n"))
    unit = match.group("u")
    if n <= 0:
        raise ValueError(f"invalid duration '{dur_str}': value must be > 0")
    if unit == "min":
        return timedelta(minutes=n)
    if unit == "h":
        return timedelta(hours=n)
    return timedelta(seconds=n)


def time_aware_downsample(series: pd.DataFrame, dur: timedelta) -> pd.DataFrame:
    """Downsample a time-indexed dataframe using fixed bins and last-value aggregation.

    Bin alignment is anchored to the floor of the first timestamp, and each bin keeps
    the **last available row** to preserve chronological causality deterministically.
    """

    if not isinstance(series.index, pd.DatetimeIndex):
        raise ValueError("time_aware_downsample requires a DateTimeIndex")
    if series.empty:
        return series.copy()

    out = series.sort_index()
    start = out.index[0].floor(pd.Timedelta(dur))
    # anchor bins to first timestamp floor to avoid calendar/clock drift.
    out = out.resample(pd.Timedelta(dur), origin=start, label="left", closed="left").last()
    out = out.dropna(how="all")
    return out
