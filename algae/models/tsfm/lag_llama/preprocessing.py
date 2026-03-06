"""
Preprocessing for Lag-Llama — time-series splits, scaling, validation.

Enforces strict time ordering, no leakage, minimum history checks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from algae.models.tsfm.lag_llama.config import LagLlamaConfig


@dataclass
class SplitResult:
    """Time-series safe train/val/test split."""
    train: pd.Series
    val: pd.Series
    test: pd.Series
    train_mean: float
    train_std: float


def validate_series(series: pd.Series, config: LagLlamaConfig) -> None:
    """Fail-fast validation on input series."""
    if series is None or len(series) == 0:
        raise ValueError("Series is empty or None")
    if len(series) < config.min_history_days:
        raise ValueError(
            f"Insufficient history: {len(series)} < {config.min_history_days} required"
        )
    if series.isna().all():
        raise ValueError("Series is all NaN")
    if not series.index.is_monotonic_increasing:
        raise ValueError("Series index is not monotonically increasing — potential leakage")


def time_series_split(
    series: pd.Series,
    config: LagLlamaConfig,
) -> SplitResult:
    """Split series into train/val/test by date fraction.

    No shuffling. Strict time ordering.
    """
    validate_series(series, config)

    n = len(series)
    train_end = int(n * config.train_pct)
    val_end = train_end + int(n * config.val_pct)

    train = series.iloc[:train_end]
    val = series.iloc[train_end:val_end]
    test = series.iloc[val_end:]

    # Compute train-only stats
    train_clean = train.dropna()
    train_mean = float(train_clean.mean()) if len(train_clean) > 0 else 0.0
    train_std = float(train_clean.std()) if len(train_clean) > 1 else 1.0
    if train_std <= 0:
        train_std = 1.0

    return SplitResult(
        train=train,
        val=val,
        test=test,
        train_mean=train_mean,
        train_std=train_std,
    )


def standardize(
    series: pd.Series,
    mean: float,
    std: float,
) -> pd.Series:
    """Z-score standardization using provided stats (typically train-only)."""
    return ((series - mean) / std).rename(series.name)


def inverse_standardize(
    series: pd.Series,
    mean: float,
    std: float,
) -> pd.Series:
    """Reverse z-score standardization."""
    return (series * std + mean).rename(series.name)
