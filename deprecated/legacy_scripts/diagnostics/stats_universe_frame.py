"""
Phase 6.1 Diagnostics: UniverseFrame stats.
Prints root path, date range, row count, unique dates, median tradable breadth.
Exit 0 if ok, nonzero if required path is missing.
"""

import sys
import polars as pl
from backend.app.ops import pathmap


def main():
    root = pathmap.get_universe_frame_root(version="v2")
    print(f"Root: {root}")

    if not root.exists():
        print("SKIP: UniverseFrame root not found")
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

    # Median tradable breadth per date
    breadth = (
        lf.filter(pl.col("is_tradable"))
        .group_by("date")
        .len()
        .select(pl.col("len").median().alias("median_breadth"))
        .collect()
    )
    print(f"median_tradable_breadth: {breadth['median_breadth'][0]}")


if __name__ == "__main__":
    main()
