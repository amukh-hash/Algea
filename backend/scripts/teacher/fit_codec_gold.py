#!/usr/bin/env python3
"""
Phase 0.5: Fit Codec Quantiles on daily equity training data.
Output: backend/models/codec/codec_daily_v1.json
"""

import os
import sys
import argparse
from pathlib import Path
import json

import numpy as np
try:
    import polars as pl
    import torch
except ImportError:
    print("Polars/Torch missing.")
    sys.exit(1)

# Import Codec
sys.path.append(os.getcwd())
try:
    from backend.app.models.chronos2_codec import Chronos2Codec, CodecConfig
except ImportError:
    print("Could not import codec.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k_files", type=int, default=10)
    args = parser.parse_args()
    
    # Config
    gold_dir = Path(os.getenv("GOLD_DAILY_PARQUET_DIR", "backend/data_canonical/daily_parquet")).resolve()
    gold_glob = os.getenv("GOLD_EXAMPLE_GLOB", "*.parquet")
    required_cols = os.getenv(
        "GOLD_REQUIRED_COLS",
        "date,open_adj,high_adj,low_adj,close_adj,volume,"
        "spy_ret_1d,qqq_ret_1d,iwm_ret_1d,vix_level,rate_proxy,market_breadth_ad"
    ).split(",")
    required_cols = [c.strip() for c in required_cols]
    
    # Exclude date column for fitting
    feat_cols = [c for c in required_cols if c != "date"]
    
    # Scan files
    files = sorted(list(gold_dir.glob(gold_glob)))
    if not files:
        print("No files found.")
        sys.exit(1)
        
    sample = files[:args.k_files]
    print(f"Scanning {len(sample)} files for fitting...")
    
    # Accumulate Windows
    # We need [Windows, Time, Features] structure for proper scaling
    # We'll use the stride logic from Phase 1 defaults to grab valid windows
    
    context = int(os.getenv("GOLD_CONTEXT", "1024"))
    pred = int(os.getenv("GOLD_PRED", "64"))
    stride = int(os.getenv("GOLD_STRIDE", "60"))
    max_wins = 500
    
    all_windows = []
    
    for f in sample:
        try:
            # Robust Read: Read all, then select.
            df = pl.read_parquet(f)
            missing = [c for c in required_cols if c not in df.columns]
            if missing:
                print(f"Skipping {f.name}: Missing {missing}. Found: {df.columns}")
                continue
                
            arr = df.select(feat_cols).to_numpy()
            n = len(arr)
            
            max_start = n - (context + pred)
            if max_start <= 0: continue
            
            starts = list(range(0, max_start, stride))
            if len(starts) > max_wins:
                 starts = np.random.choice(starts, max_wins, replace=False)
                 
            for s in starts:
                w = arr[s : s + context] # Use context for fitting distribution
                all_windows.append(w)
                
        except Exception as e:
            print(f"Err {f}: {e}")
            
    if not all_windows:
        print("No valid windows found.")
        sys.exit(1)
        
    # Stack: [N, T, F]
    data = np.stack(all_windows)
    print(f"Fitting on shape {data.shape}...")
    
    # Fit
    codec = Chronos2Codec(CodecConfig(vocab_size=4096))
    codec.fit_quantiles(data, feature_names=feat_cols)
    
    # Save
    out_path = Path("backend/models/codec/codec_daily_v1.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    codec.save(out_path)
    print(f"Saved artifact to {out_path}")

if __name__ == "__main__":
    main()
