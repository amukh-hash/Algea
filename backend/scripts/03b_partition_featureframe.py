"""
Script: 03b_partition_featureframe.py
Purpose: Partition values in FeatureFrame into per-ticker parquet files for sequential training.
Input: backend/features/featureframe_vv1.parquet
Output: backend/data_canonical/daily_parquet/{ticker}.parquet
"""
import os
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from backend.app.ops import pathmap, bootstrap

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="backend/data_canonical/daily_parquet")
    parser.add_argument("--featureframe_tag", default="vv1")
    args = parser.parse_args()

    bootstrap.ensure_dirs()
    
    # Resolve input (Direct)
    ff_path = os.path.abspath("backend/features/featureframe_vv1.parquet")
    
    if not os.path.exists(ff_path):
        # Validation
        raise FileNotFoundError(f"FeatureFrame not found at {ff_path}")
    
    logger.info(f"Reading FeatureFrame from {ff_path}...")
    df = pd.read_parquet(ff_path)
    
    # Ensure sorted by date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")
    
    # Output Dir
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    
    logger.info(f"Partitioning {len(df)} rows into {out_dir}...")
    
    # NEW: Filter by Universe Manifest(s)
    # We load all available manifests to build a "Training Universe" (Union of eligible)
    # Or just use the symbols present in the manifests.
    manifest_root = "backend/manifests" # Correct path
    # Resolve valid symbols
    valid_symbols = set()
    if os.path.exists(manifest_root):
        import glob
        files = glob.glob(os.path.join(manifest_root, "*.parquet"))
        logger.info(f"Found {len(files)} manifests. Building allowed symbol list...")
        for f in files:
            try:
                mdf = pd.read_parquet(f)
                if "eligible" in mdf.columns:
                    eligible = mdf[mdf["eligible"]]["symbol"].tolist()
                    valid_symbols.update(eligible)
                elif "is_eligible" in mdf.columns:
                     eligible = mdf[mdf["is_eligible"]]["symbol"].tolist()
                     valid_symbols.update(eligible)
            except Exception as e:
                logger.warning(f"Failed to read manifest {f}: {e}")
    
    if valid_symbols:
        logger.info(f"Filtering FeatureFrame to {len(valid_symbols)} / {df['symbol'].nunique()} unique symbols.")
        df = df[df["symbol"].isin(valid_symbols)]
    else:
        logger.warning("No universe manifests found (or empty). Partitioning ALL symbols (Legacy behavior).")

    if df.empty:
        logger.error("FeatureFrame is empty after filtering! Check universe manifests.")
        return
        
    grouped = df.groupby("symbol")
    count = 0
    
    for symbol, group in tqdm(grouped):
        # Sort again to be safe
        group = group.sort_values("date").reset_index(drop=True)
        
        # Write
        out_path = os.path.join(out_dir, f"{symbol}.parquet")
        group.to_parquet(out_path)
        count += 1
        
    logger.info(f"Partitioned {count} tickers.")

if __name__ == "__main__":
    main()
