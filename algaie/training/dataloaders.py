from __future__ import annotations

from typing import Iterable, Iterator, Tuple

import pandas as pd


def iter_batches(df: pd.DataFrame, batch_size: int) -> Iterator[pd.DataFrame]:
    for start in range(0, len(df), batch_size):
        yield df.iloc[start : start + batch_size]


def split_train_test(df: pd.DataFrame, split_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_ts = pd.Timestamp(split_date)
    train = df[df["date"] <= split_ts].copy()
    test = df[df["date"] > split_ts].copy()
    return train, test
