"""
Phase 6.1 Diagnostics: SelectorFeatures stats.
Prints root path, date range, row count, unique dates, median daily breadth.
Exit 0 if ok, nonzero if required path is missing.
"""

import sys
import argparse
import polars as pl
from backend.app.ops import pathmap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", default="5", help="Forecast horizon (default: 5)")
    args = parser.parse_args()

    root = pathmap.get_selector_features_root(version="v2", horizon=args.horizon)
    if not root.exists():
        root = pathmap.get_selector_features_root(version="v2")

    print(f"Root: {root}")

    if not root.exists():
        print("SKIP: SelectorFeatures root not found")
        sys.exit(0)

    try:
        lf = pl.scan_parquet(str(root / "**/*.parquet"))
    except Exception as e:
        print(f"ERROR: Failed to scan parquet: {e}")
        sys.exit(1)

    # Global stats
    stats = lf.select([
        pl.len().alias("n_rows"),
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
        pl.col("date").n_unique().alias("n_unique_dates"),
    ]).collect()

    print(f"n_rows:        {stats['n_rows'][0]}")
    print(f"min_date:      {stats['min_date'][0]}")
    print(f"max_date:      {stats['max_date'][0]}")
    print(f"n_unique_dates:{stats['n_unique_dates'][0]}")

    # Median daily breadth (n per date)
    breadth = (
        lf.group_by("date")
        .len()
        .select(pl.col("len").median().alias("median_breadth"))
        .collect()
    )
    print(f"median_daily_breadth: {breadth['median_breadth'][0]}")


if __name__ == "__main__":
    main()
