import polars as pl
import pandas as pd
import numpy as np
import torch
from typing import List, Tuple, Optional, Dict
from datetime import datetime, timedelta
import os

def make_cross_sectional_batch(
    target_date: datetime.date,
    universe: List[str],
    data_dir: str,
    breadth_path: str,
    lookback_days: int = 60,
    horizon_days: int = 10,
    feature_cols: List[str] = None,
    embargo_days: int = 10,
    purge_days: int = 5
) -> Dict[str, torch.Tensor]:
    """
    Creates a cross-sectional batch for a single date.
    X: [NumTickers, Lookback, Features]
    y: [NumTickers] (Forward 10d return)

    Filters:
    - Tickers must have full history for lookback.
    - Tickers must have forward label (unless inference mode).
    - Embargo/Purge logic applied implicitly by ensuring train/test splits don't overlap.
      (This function just builds the batch for a given date).
    """
    if feature_cols is None:
        raise ValueError("Must provide feature_cols list")
        
    batch_X = []
    batch_y = []
    valid_tickers = []

    # Load breadth once
    try:
        breadth_df = pl.read_parquet(breadth_path)
    except FileNotFoundError:
        print(f"Breadth file not found: {breadth_path}")
        return None
        
    for ticker in universe:
        try:
            # Load Ticker Data
            # Assume we have a helper to load processed features + priors attached
            # Or we load raw and process on fly?
            # Ideally: We load precomputed features.
            # But for training loop efficiency, usually we load a big parquet or database.
            # Let's assume we load a parquet file per ticker that has features + targets.
            
            # Path: backend/data/features/{ticker}.parquet
            fpath = f"{data_dir}/features/{ticker}.parquet"
            if not os.path.exists(fpath):
                continue
                
            df = pl.read_parquet(fpath)

            # Filter to relevant window
            # We need [target_date - lookback, target_date] for X
            # And [target_date, target_date + horizon] for y

            # Ensure date col
            if "date" not in df.columns:
                df = df.with_columns(pl.col("timestamp").cast(pl.Date).alias("date"))
                
            # Get index of target date
            # We need exact row? Or closest?
            # Strict: Must exist.
            target_row = df.filter(pl.col("date") == target_date)

            if len(target_row) == 0:
                continue
                
            # We need previous lookback rows
            # Sort just in case
            df = df.sort("date")

            # Find row index
            # Polars doesn't have integer index easily without with_row_count
            df = df.with_row_count("idx")
            target_idx = df.filter(pl.col("date") == target_date)["idx"].item()

            start_idx = target_idx - lookback_days + 1
            if start_idx < 0:
                continue
                
            # Slice X
            # [start_idx, target_idx] inclusive (length = lookback)
            # Ensure we take exactly lookback rows
            slice_x = df.slice(start_idx, lookback_days)

            if len(slice_x) != lookback_days:
                continue
                
            # Extract Features
            x_vals = slice_x.select(feature_cols).to_numpy() # [L, F]

            # Extract Target
            # 10-day forward return
            # We need row at target_idx + horizon
            target_future_idx = target_idx + horizon_days

            if target_future_idx >= len(df):
                continue
                
            # Return = (Price[t+h] / Price[t]) - 1
            # Or log return diff?
            # Let's use simple return for ranking? Or log return?
            # "log_return_10d_fwd" might be precomputed.
            
            # Calculate on fly if needed
            p_t = df[target_idx, "close"]
            p_fut = df[target_future_idx, "close"]

            if p_t is None or p_fut is None or p_t == 0:
                continue
                
            ret = (p_fut / p_t) - 1.0

            # Aux target: Direction (1 if ret > 0 else 0)
            direction = 1.0 if ret > 0 else 0.0

            batch_X.append(x_vals)
            batch_y.append([ret, direction])
            valid_tickers.append(ticker)

        except Exception as e:
            # print(f"Error processing {ticker}: {e}")
            continue

    if not batch_X:
        return None
        
    # Stack
    # X: [B, T, F]
    X_tensor = torch.tensor(np.stack(batch_X), dtype=torch.float32)

    # y: [B, 2] (Return, Direction)
    y_tensor = torch.tensor(np.stack(batch_y), dtype=torch.float32)

    return {
        "X": X_tensor,
        "y": y_tensor[:, 0], # Primary target: return
        "y_aux": y_tensor[:, 1], # Aux target: direction
        "tickers": valid_tickers
    }
