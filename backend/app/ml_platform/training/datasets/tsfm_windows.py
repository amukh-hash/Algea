"""Time-aware TSFM window builder utilities."""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from ...utils.downsample import parse_duration_to_timedelta, time_aware_downsample


def _normalize_timestamps(timestamps: list[str] | list[datetime]) -> list[pd.Timestamp]:
    return [pd.Timestamp(ts) for ts in timestamps]


def downsample_series_time_aware(
    series: list[float],
    timestamps: list[str] | list[datetime],
    downsample_freq: str = "1min",
) -> tuple[list[float], list[pd.Timestamp]]:
    """Downsample value series using explicit timestamps and duration string frequency."""

    if len(series) != len(timestamps):
        raise ValueError("series and timestamps length mismatch")
    dur = parse_duration_to_timedelta(downsample_freq)
    idx = _normalize_timestamps(timestamps)
    frame = pd.DataFrame({"value": [float(v) for v in series]}, index=pd.DatetimeIndex(idx))
    out = time_aware_downsample(frame, dur)
    return out["value"].astype(float).tolist(), list(out.index)


def build_tsfm_windows(
    series: list[float],
    context_length: int,
    prediction_length: int,
    timestamps: list[str] | list[datetime] | None = None,
    downsample_freq: str = "1min",
) -> list[tuple[list[float], list[float]]]:
    windows: list[tuple[list[float], list[float]]] = []

    values = [float(v) for v in series]
    if timestamps is not None:
        values, _ = downsample_series_time_aware(values, timestamps, downsample_freq=downsample_freq)

    span = context_length + prediction_length
    if len(values) < span:
        return windows
    for i in range(0, len(values) - span + 1):
        chunk = values[i : i + span]
        windows.append((chunk[:context_length], chunk[context_length:]))
    return windows
