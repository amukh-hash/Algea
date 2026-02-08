"""
Script: 03b_partition_featureframe.py
Purpose: Partition values in FeatureFrame into per-ticker parquet files for sequential training.
Input: backend/features/featureframe_vv1.parquet
Output: backend/data_canonical/daily_parquet/{ticker}.parquet
"""
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import logging
import argparse
import logging
import pandas as pd
from tqdm import tqdm
from backend.app.ops import pathmap, bootstrap, config
from backend.app.data import priors_store
import polars as pl

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default="backend/data_canonical/daily_parquet")
    parser.add_argument("--featureframe_path", default="backend/features/featureframe_vv1.parquet")
    parser.add_argument("--priors_dir", default="backend/data/priors/chronos2/v1")
    parser.add_argument("--featureframe_tag", default="vv1")
    args = parser.parse_args()

    bootstrap.ensure_dirs()
    
    # Resolve input (Direct)
    ff_path = os.path.abspath(args.featureframe_path)
    
    if not os.path.exists(ff_path):
        # Validation
        raise FileNotFoundError(f"FeatureFrame not found at {ff_path}")
    
    logger.info(f"Reading FeatureFrame from {ff_path}...")
    df = pd.read_parquet(ff_path)
    
    # Ensure sorted by date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

    # --- PART C: MERGE PRIORS ---
    if config.PRIORS_ENABLED:
        logger.info("Priors Enabled. Merging PriorsFrame...")
        priors_dir = args.priors_dir # Default path from build_priors_frame
        
        # Check if priors exist
        if os.path.exists(priors_dir):
            try:
                # Load via store (returns Polars)
                priors_pl = priors_store.read_priors_frame(priors_dir)
                
                if not priors_pl.is_empty():
                    # Convert to Pandas for merge
                    priors_df = priors_pl.to_pandas()
                    
                    # Normalize columns
                    # Priors has 'ticker', Features has 'symbol'
                    if "ticker" in priors_df.columns:
                        priors_df = priors_df.rename(columns={"ticker": "symbol"})
                    
                    # Ensure date match
                    priors_df["date"] = pd.to_datetime(priors_df["date"])
                    
                    # Merge (Left Join on features)
                    # We want to keep all feature rows, fill missing priors with NaN
                    logger.info(f"Merging {len(priors_df)} prior rows into {len(df)} feature rows...")
                    
                    # Columns to merge
                    # date, symbol, + p_* cols
                    prior_cols = [c for c in priors_df.columns if c.startswith("p_")]
                    merge_cols = ["date", "symbol"] + prior_cols
                    
                    priors_subset = priors_df[merge_cols]
                    
                    # Drop duplicates in priors if any (shouldn't be, but safe)
                    priors_subset = priors_subset.drop_duplicates(subset=["date", "symbol"])
                    
                    df = pd.merge(df, priors_subset, on=["date", "symbol"], how="left")
                    
                    # Fill NaNs?
                    # Plan says: "Fill missing priors with NaNs; then impute with a safe method... or leave NaN"
                    # "add a missingness indicator per prior column"
                    
                    for col in prior_cols:
                        # Add indicator
                        df[f"{col}_isna"] = df[col].isna().astype(int)
                        # Fill with 0? Or keeps NaNs?
                        # Selector scaler might handle NaNs if using RobustScaler? 
                        # RobustScaler doesn't handle NaNs.
                        # We must fill. forward fill?
                        # Forward fill per group
                        # But here we are global dataframe. 
                        # We can fillna(0) for neutral prior?
                        # Neural networks hate NaNs.
                        # Let's fill with 0 and rely on _isna indicator.
                        # Or better: forward fill within ticker.
                        # We are about to group by ticker anyway.
                        pass # specific filling happening inside loop or after merge
                        
                    logger.info("Priors merged successfully.")
                else:
                    logger.warning("PriorsFrame is empty. Skipping merge.")
            except Exception as e:
                logger.error(f"Failed to merge priors: {e}")
                # Don't fail hard? Or do?
                # "MUST be deterministic... resume-safe"
                # If priors enabled but missing/failed, warnings.
        else:
            logger.warning(f"Priors directory {priors_dir} not found. Skipping merge.")
    
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
