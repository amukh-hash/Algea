from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd


@dataclass(frozen=True)
class WalkForwardSplit:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def build_walk_forward_splits(
    dates: pd.DatetimeIndex,
    train_window_days: int,
    test_window_days: int,
    step_days: int,
    expanding: bool,
    holdout_pct: float,
) -> List[WalkForwardSplit]:
    dates = pd.DatetimeIndex(sorted(dates.unique()))
    holdout_size = int(len(dates) * holdout_pct)
    train_end_limit = len(dates) - holdout_size
    splits: List[WalkForwardSplit] = []
    start_idx = 0
    while True:
        train_end = start_idx + train_window_days
        test_end = train_end + test_window_days
        if test_end > train_end_limit:
            break
        train_start_idx = 0 if expanding else start_idx
        splits.append(
            WalkForwardSplit(
                train_start=dates[train_start_idx],
                train_end=dates[train_end - 1],
                test_start=dates[train_end],
                test_end=dates[test_end - 1],
            )
        )
        start_idx += step_days
    return splits
