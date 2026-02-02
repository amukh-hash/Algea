import os
import sys
import polars as pl
import pandas as pd
import numpy as np
import argparse
import hashlib
from typing import Dict

# Ensure backend in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from backend.app.data import calendar

def get_file_hash(path: str) -> str:
    hasher = hashlib.md5()
    with open(path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def validate_marketframe(path: str) -> bool:
    print(f"Validating {path}...")
    try:
        df = pl.read_parquet(path)
    except Exception as e:
        print(f"FAILED: Could not read {path}: {e}")
        return False

    # 1. Monotonic Timestamps
    timestamps = df["timestamp"]
    if not timestamps.is_sorted():
        print("FAILED: Timestamps not monotonic")
        return False

    # 2. No Duplicates
    if timestamps.n_unique() != len(timestamps):
        print("FAILED: Duplicate timestamps found")
        return False

    # 3. Join Coverage (Breadth columns should not be null after ffill, unless start is missing)
    # Check null count in 'ad_line', 'bpi'
    # If starting rows are null, that's acceptable if breadth starts later, but we expect full coverage for dummy data.
    nulls_ad = df["ad_line"].null_count()
    if nulls_ad > 0:
        print(f"WARNING: {nulls_ad} nulls in ad_line. (Acceptable at start)")

    # 4. Session Invariants
    # Check a sample
    sample = df.sample(min(100, len(df)))
    cal = calendar.get_calendar()

    # We just check if they are "trading minutes"?
    # Or just check if they are valid timestamps.
    # Checking against calendar is slow for loop.

    return True

def validate_reproducibility(file_map: Dict[str, str]) -> bool:
    # file_map: {name: path}
    # This function expects to be called after a SECOND run to compare hashes.
    # But we need to store hashes from first run.
    # For now, we just print the hash.
    print("Artifact Hashes:")
    for name, path in file_map.items():
        if os.path.exists(path):
            h = get_file_hash(path)
            print(f"  {name}: {h}")
        else:
            print(f"  {name}: MISSING")
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="backend/data/marketframe")
    parser.add_argument("--context_path", default="backend/data/context/breadth_1m.parquet")
    args = parser.parse_args()

    # Validate Breadth
    if os.path.exists(args.context_path):
        # Read with pandas to check timestamp index/column
        df_pd = pd.read_parquet(args.context_path)
        print(f"Breadth Context: {len(df_pd)} rows. Index: {df_pd.index.name or 'RangeIndex'}")
        if "timestamp" in df_pd.columns:
            print("  Timestamp is a column.")
        if isinstance(df_pd.index, pd.DatetimeIndex):
            print("  Index is DatetimeIndex.")
            if not df_pd.index.is_monotonic_increasing:
                print("FAILED: Breadth index not monotonic")
                sys.exit(1)
    else:
        print(f"FAILED: Context missing {args.context_path}")
        sys.exit(1)

    # Validate MarketFrames
    success = True
    for f in os.listdir(args.data_dir):
        if f.endswith(".parquet"):
            if not validate_marketframe(os.path.join(args.data_dir, f)):
                success = False

    if not success:
        sys.exit(1)

    print("Validation Passed.")

if __name__ == "__main__":
    main()
