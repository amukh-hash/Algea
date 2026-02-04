import os
import sys
import json
import pandas as pd
import numpy as np
from datetime import datetime
import argparse

# Ensure backend is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from backend.app.data import breadth, manifest

DATA_DIR = os.path.join("backend", "data_cache_alpaca")
OUTPUT_DIR = os.path.join("backend", "data", "context")
OUTPUT_FILE = "breadth_1m.parquet"
METADATA_FILE = "breadth_metadata.json"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DATA_DIR)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Universe
    universe = manifest.get_context_universe()
    print(f"Loading data for {len(universe)} tickers...")

    dfs = {}
    
    # Check if data dir exists
    if not os.path.exists(args.data_dir):
        print(f"WARNING: Data directory {args.data_dir} does not exist.")
        # Create dummy data for structure verification if strictly needed, 
        # but better to just exit or warn. 
        # For the purpose of 'smoke test', we might need to create dummy data externally.
        return

    # Load data
    for ticker in universe:
        fpath = os.path.join(args.data_dir, f"{ticker}_1m.parquet")
        if os.path.exists(fpath):
            try:
                # Try reading timestamp column too if it exists
                # We don't know if it's index or column.
                # Read everything or try to guess?
                # Let's read open/high/low/close and 'timestamp' if possible.
                # But read_parquet columns=... filters what is read.
                # If we omit columns, we read all.
                # Let's read all cols, it's safer for 1m data (small enough).
                df = pd.read_parquet(fpath)
                
                # If timestamp is a column, set it as index
                if "timestamp" in df.columns:
                    df = df.set_index("timestamp")
                    
                # Ensure index is unique
                if not df.index.is_unique:
                    df = df[~df.index.duplicated(keep='first')]
                dfs[ticker] = df
            except Exception as e:
                print(f"Failed to load {ticker}: {e}")
    
    if not dfs:
        print("No data loaded. Exiting.")
        return

    # 2. Unified Index
    min_date = min([df.index.min() for df in dfs.values()])
    max_date = max([df.index.max() for df in dfs.values()])
    master_index = pd.date_range(min_date, max_date, freq='1min', tz="UTC") # Assuming UTC or naive?
    # Alpaca is usually UTC. Let's assume input dfs are tz-aware UTC or compatible.
    # If dfs are naive, master_index should be naive.
    # Check first df
    first_df = next(iter(dfs.values()))
    if first_df.index.tz is not None:
         master_index = pd.date_range(min_date, max_date, freq='1min', tz=first_df.index.tz)

    print(f"Building breadth for range {min_date} to {max_date}")

    # 3. Compute AD Line
    print("Computing AD Line...")
    closes = pd.DataFrame(index=master_index)
    for t, df in dfs.items():
        closes[t] = df['close'].reindex(master_index).ffill(limit=5)
    
    ad_line = breadth.calculate_ad_line(closes)

    # 4. Compute BPI
    print("Computing BPI...")
    bpi = breadth.calculate_bpi(dfs, master_index)

    # 5. Save
    out_df = pd.DataFrame({
        'ad_line': ad_line,
        'bpi': bpi
    }, index=master_index)
    
    # Save Parquet with explicit timestamp column
    out_df = out_df.reset_index().rename(columns={"index": "timestamp"})
    
    out_path = os.path.join(args.output_dir, OUTPUT_FILE)
    out_df.to_parquet(out_path, index=False)
    print(f"Saved breadth data to {out_path}")

    # 6. Save Metadata
    meta = {
        "universe_hash": manifest.get_microcosm_hash(),
        "universe_size": len(universe),
        "start_date": str(min_date),
        "end_date": str(max_date),
        "fill_policy": "ffill_limit_5",
        "version": "1.0",
        "timestamp": str(datetime.now())
    }
    with open(os.path.join(args.output_dir, METADATA_FILE), 'w') as f:
        json.dump(meta, f, indent=2)
    print("Saved metadata.")

if __name__ == "__main__":
    main()
