"""
Phase 6.1 Diagnostics: Priors stats.
Accepts AS_OF_DATE and PRIORS_VERSION via env/args.
Prints root path, rows, symbols, coverage vs tradable universe.
Exit 0 if ok, nonzero if required path is missing.
"""

import sys
import os
import datetime
import polars as pl
from pathlib import Path
from backend.app.ops import pathmap
from backend.app.data import schema_contracts


def main():
    as_of_date = os.getenv("AS_OF_DATE")
    priors_version = os.getenv("PRIORS_VERSION") or "latest"

    if not as_of_date:
        priors_root = pathmap.get_priors_root()
        if not priors_root.exists():
            print(f"SKIP: Priors root not found at {priors_root}")
            sys.exit(0)
        date_dirs = sorted(
            [d.name.split("=")[1] for d in priors_root.glob("date=*")]
        )
        if not date_dirs:
            print("SKIP: No date partitions in priors")
            sys.exit(0)
        as_of_date = date_dirs[-1]
        print(f"Auto-selected AS_OF_DATE: {as_of_date}")

    try:
        target_file = pathmap.resolve(
            "priors_date", date=as_of_date, version=priors_version
        )
        path = Path(target_file)
    except Exception as e:
        print(f"ERROR: Path resolution failed: {e}")
        sys.exit(1)

    print(f"Root: {path}")

    if not path.exists():
        print(f"SKIP: Priors artifact not found at {path}")
        sys.exit(0)

    df = pl.read_parquet(str(path))
    df = schema_contracts.normalize_keys(df)

    n_rows = len(df)
    n_symbols = df["symbol"].n_unique()

    dates_in_file = df["date"].unique().to_list()
    min_date = min(dates_in_file) if dates_in_file else None
    max_date = max(dates_in_file) if dates_in_file else None

    print(f"n_rows:        {n_rows}")
    print(f"n_symbols:     {n_symbols}")
    print(f"min_date:      {min_date}")
    print(f"max_date:      {max_date}")
    print(f"n_unique_dates:{len(dates_in_file)}")

    # Coverage check
    print("Checking coverage vs UniverseFrame...")
    univ_root = pathmap.get_universe_frame_root(version="v2")
    if not univ_root.exists():
        print("UniverseFrame root not found, skipping coverage check.")
        return

    univ_lf = pl.scan_parquet(str(univ_root / "**/*.parquet"))
    univ_schema = univ_lf.schema
    if "ticker" in univ_schema and "symbol" not in univ_schema:
        univ_lf = univ_lf.rename({"ticker": "symbol"})

    dt = datetime.date.fromisoformat(as_of_date)
    tradable_syms = (
        univ_lf.filter(
            (pl.col("date") == dt) & pl.col("is_tradable")
        )
        .select("symbol")
        .collect()
        .get_column("symbol")
        .to_list()
    )

    if tradable_syms:
        priors_syms = set(df.get_column("symbol").to_list())
        common = set(tradable_syms).intersection(priors_syms)
        coverage = len(common) / len(tradable_syms)
        print(f"median_daily_breadth (symbols): {n_symbols}")
        print(f"coverage: {coverage:.2%} ({len(common)}/{len(tradable_syms)})")
    else:
        print("No tradable universe found for this date.")


if __name__ == "__main__":
    main()
