from __future__ import annotations

import pandas as pd


def purged_embargo_splits(df: pd.DataFrame, fold: int, embargo: int) -> list[tuple[pd.Index, pd.Index]]:
    splits = []
    for start in range(0, len(df) - fold, fold):
        test_idx = df.index[start : start + fold]
        train_mask = ~df.index.isin(df.index[max(0, start - embargo) : start + fold + embargo])
        splits.append((df.index[train_mask], test_idx))
    return splits
