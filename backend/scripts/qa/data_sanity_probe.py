#!/usr/bin/env python3
"""
Quick data sanity probe for daily_parquet sources.
Loads a few random parquet files and prints target stats.
"""

import os
import random
import pandas as pd
from backend.app.ops import pathmap


def main():
    paths = pathmap.get_paths()
    daily_root = os.path.join(paths.data_canonical, "daily_parquet")
    if not os.path.exists(daily_root):
        raise FileNotFoundError(f"Missing daily_parquet root: {daily_root}")

    candidates = [
        os.path.join(daily_root, f)
        for f in os.listdir(daily_root)
        if f.endswith(".parquet")
    ]
    if not candidates:
        raise FileNotFoundError(f"No parquet files found in {daily_root}")

    sample = random.sample(candidates, k=min(5, len(candidates)))
    target_col = "ret_1d"

    for path in sample:
        df = pd.read_parquet(path)
        print(f"\nFile: {path}")
        print(f"Columns: {sorted(df.columns)}")
        if target_col not in df.columns:
            print(f"Missing target column: {target_col}")
            continue
        series = pd.to_numeric(df[target_col], errors="coerce")
        total = len(series)
        zeros = (series == 0).sum()
        nans = series.isna().sum()
        print(
            f"{target_col}: min={series.min():.6f} max={series.max():.6f} "
            f"mean={series.mean():.6f} std={series.std(ddof=0):.6f}"
        )
        print(f"{target_col}: zeros={zeros/total:.2%} nans={nans/total:.2%}")


if __name__ == "__main__":
    main()
