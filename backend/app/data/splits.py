import polars as pl
from datetime import datetime, timedelta
from typing import Tuple, List, Iterator

def get_time_splits(df: pl.DataFrame, val_months: int = 6, test_months: int = 6) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Splits DataFrame into Train, Val, Test based on time.
    Assumes df is sorted by timestamp.
    Test is the last `test_months`.
    Val is the `val_months` before Test.
    Train is everything before.
    """
    max_date = df.select(pl.max("timestamp")).item()

    test_start = max_date - timedelta(days=30*test_months)
    val_start = test_start - timedelta(days=30*val_months)

    train = df.filter(pl.col("timestamp") < val_start)
    val = df.filter((pl.col("timestamp") >= val_start) & (pl.col("timestamp") < test_start))
    test = df.filter(pl.col("timestamp") >= test_start)

    return train, val, test

def rolling_holdout_split(df: pl.DataFrame, window_size_days: int = 365, step_days: int = 30) -> Iterator[Tuple[pl.DataFrame, pl.DataFrame]]:
    """
    Yields (train, val) pairs for rolling evaluation.
    Train is fixed window size? Or expanding?
    Usually expanding train, rolling val.
    Let's assume Expanding Train.
    """
    min_date = df.select(pl.min("timestamp")).item()
    max_date = df.select(pl.max("timestamp")).item()

    current_date = min_date + timedelta(days=window_size_days)

    while current_date < max_date:
        val_end = current_date + timedelta(days=step_days)

        train = df.filter(pl.col("timestamp") < current_date)
        val = df.filter((pl.col("timestamp") >= current_date) & (pl.col("timestamp") < val_end))

        if val.height > 0:
            yield train, val

        current_date = val_end
