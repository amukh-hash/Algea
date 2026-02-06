
import sys
import os
import torch
import numpy as np
from pathlib import Path

# Fix import path
sys.path.append(os.getcwd())

from backend.scripts.teacher.phase1_train_teacher_gold import GoldFuturesWindowDataset

# Mock config
class MockConfig:
    gold_parquet_dir = Path("backend/data_canonical/daily_parquet")
    gold_glob = "*.parquet"
    required_cols = tuple("date,open_adj,high_adj,low_adj,close_adj,volume,spy_ret_1d,qqq_ret_1d,iwm_ret_1d,vix_level,rate_proxy,market_breadth_ad".split(","))
    context = 1024
    pred = 10
    stride_rows = 60
    max_files_limit = 5
    cache_size = 5
    target_col = "close_adj"

cfg = MockConfig()
files = sorted(cfg.gold_parquet_dir.glob(cfg.gold_glob))[:cfg.max_files_limit]

print(f"Loading {len(files)} files...")
ds = GoldFuturesWindowDataset(
    files=files,
    required_cols=cfg.required_cols,
    context=cfg.context,
    pred=cfg.pred,
    stride_rows=cfg.stride_rows, 
    max_windows=100, 
    seed=42, 
    cache_size=cfg.cache_size,
    target_col=cfg.target_col
)

train_ds, val_ds = ds.split_validation(0.15)
print(f"Train size: {len(train_ds)}, Val size: {len(val_ds)}")

# Check Train Sample (index 0 might be problematic if dropped?)
# Let's try a valid index
if len(train_ds) > 0:
    t_idx = 0
    t_sample = train_ds[t_idx]
    # Check shape
    x = t_sample["x_float"]
    y = t_sample["y_float"]
    # Identify target col dim
    # Dataset logic: returns subset of cols if ts excluded
    # Inline dataset logic: 
    # if ts_idx not None: feats = subset[:, feat_indices]
    # target_feat_idx calculated dynamically.
    
    # We just want to check the VALUES of the output array.
    # We don't verify WHICH column is target here, but if the whole array is prices (150.0) or returns (0.01).
    
    print(f"Train X Mean/Max: {x.mean():.4f} / {x.max():.4f}")
    
    # Check specific target column (Index 3: close_adj)
    # feats has removed date. required=date,open,high,low,close...
    # feats=open,high,low,close...
    # close is index 3.
    target_sample = y[:, 3]
    print(f"Train Y (Target) Mean/Max: {target_sample.mean():.4f} / {target_sample.max():.4f}")

if len(val_ds) > 0:
    v_idx = 0
    v_sample = val_ds[v_idx]
    vx = v_sample["x_float"]
    vy = v_sample["y_float"]
    print(f"Val X Mean/Max: {vx.mean():.4f} / {vx.max():.4f}")
    
    val_target = vy[:, 3]
    print(f"Val Y (Target) Mean/Max: {val_target.mean():.4f} / {val_target.max():.4f}")

    if np.abs(val_target).max() > 10.0:
        print(f"FAIL: Validation Target Max is {np.abs(val_target).max()} (Raw Prices?)")
    else:
        print(f"SUCCESS: Validation Target Max is {np.abs(val_target).max()} (Log Returns).")
