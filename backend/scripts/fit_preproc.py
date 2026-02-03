import os
import sys
import polars as pl
import argparse
from datetime import datetime
import pandas as pd

# Ensure backend in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from backend.app.preprocessing import preproc
from backend.app.data import manifest

MARKETFRAME_DIR = os.path.join("backend", "data", "marketframe")
OUTPUT_DIR = os.path.join("backend", "models", "preproc")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v1")
    parser.add_argument("--start_date", default="2020-01-01")
    parser.add_argument("--end_date", default="2023-01-01") # Training split end
    parser.add_argument("--data_dir", default=MARKETFRAME_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tickers = manifest.get_trade_universe()
    print(f"Loading data for {len(tickers)} tickers to fit preprocessor...")

    dfs = []

    # We don't need ALL data to fit mean/std. A random sample or subset is fine.
    # But for reproducibility, we use the defined range.

    for ticker in tickers:
        fpath = os.path.join(args.data_dir, f"marketframe_{ticker}_1m.parquet")
        if os.path.exists(fpath):
            try:
                # Load lazily to filter?
                # Polars is fast.
                df = pl.read_parquet(fpath)

                # Filter dates
                # Ensure start_date/end_date are timezone aware if column is UTC
                start_dt = datetime.fromisoformat(args.start_date)
                end_dt = datetime.fromisoformat(args.end_date)

                # If naive, assume UTC
                if start_dt.tzinfo is None:
                    start_dt = start_dt.replace(tzinfo=datetime.now().astimezone().tzinfo).astimezone(None) # local to utc? Or just force UTC?
                    # Our dummy data is UTC.
                    # Let's just use strings for Polars comparison if possible, or cast.

                # Polars handles datetime comparison best if we use pl.lit() or proper datetimes.
                # The error suggests precision mismatch ns vs us.

                # We cast column to us (microseconds) to match python datetime usually, or cast python to ns.
                # But safer: convert input string to UTC datetime (python) and polars should handle it if timezone matches.

                start_dt = datetime.fromisoformat(args.start_date).replace(tzinfo=datetime.now().astimezone().tzinfo).astimezone(None) # UTC
                end_dt = datetime.fromisoformat(args.end_date).replace(tzinfo=datetime.now().astimezone().tzinfo).astimezone(None) # UTC

                # Actually, simpler:
                # Our parquet has UTC timestamps.
                # We can just compare against UTC datetime objects.
                # fromisoformat might be naive.

                s = datetime.fromisoformat(args.start_date)
                # Ensure UTC
                if s.tzinfo is None:
                     s = s.replace(tzinfo=pd.Timestamp.now().tz).astimezone(pd.Timestamp.now(tz="UTC").tz)
                else:
                     s = s.astimezone(pd.Timestamp.now(tz="UTC").tz)

                e = datetime.fromisoformat(args.end_date)
                if e.tzinfo is None:
                     e = e.replace(tzinfo=pd.Timestamp.now().tz).astimezone(pd.Timestamp.now(tz="UTC").tz)
                else:
                     e = e.astimezone(pd.Timestamp.now(tz="UTC").tz)

                # Try to use polars literal without casting the column (preserve its type)
                # But ensure literal matches.
                # If column is Naive, comparison with UTC literal fails.
                # If column is UTC, comparison with Naive literal fails.

                # Check column type? Lazy -> can't check easily without schema.
                # Eager read (we did pl.read_parquet).
                # Check schema.
                ts_dtype = df.schema["timestamp"]

                # If Naive, make literal Naive
                if isinstance(ts_dtype, pl.Datetime) and ts_dtype.time_zone is None:
                    s = s.replace(tzinfo=None)
                    e = e.replace(tzinfo=None)

                df = df.filter(
                    (pl.col("timestamp") >= s) &
                    (pl.col("timestamp") < e)
                )

                if df.height > 0:
                    dfs.append(df)
            except Exception as e:
                print(f"Error loading {ticker}: {e}")

    if not dfs:
        print("No data found for fitting. Exiting.")
        return

    print("Concatenating...")
    full_df = pl.concat(dfs)

    print(f"Fitting on {full_df.height} rows...")
    p = preproc.Preprocessor()
    p.fit(full_df)

    out_path = os.path.join(args.output_dir, f"preproc_{args.version}.json")
    p.save(out_path)
    print(f"Saved preprocessor to {out_path}")
    print(f"Version Hash: {p.version_hash}")
    print("Params:", p.params)

if __name__ == "__main__":
    main()
