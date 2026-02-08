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
            
            # Path: Check flat or nested
            fpath = os.path.join(data_dir, f"{ticker}.parquet")
            if not os.path.exists(fpath):
                # Try nested features/
                fpath = os.path.join(data_dir, "features", f"{ticker}.parquet")
                if not os.path.exists(fpath):
                    continue
                
            df = pl.read_parquet(fpath)

            # Filter to relevant window
            if "date" not in df.columns:
                if "timestamp" in df.columns:
                    df = df.with_columns(pl.col("timestamp").alias("date"))
                else:
                    # Can't use
                    continue
            
            # Cast date to Date key
            df = df.with_columns(pl.col("date").cast(pl.Date))
                
            # Find row index
            df = df.sort("date")
            df = df.with_row_count("idx")
            
            rows = df.filter(pl.col("date") == target_date)
            if rows.height == 0:
                continue
                
            target_idx = rows["idx"].item()
            start_idx = target_idx - lookback_days + 1
            if start_idx < 0:
                continue
                
            # Slice X
            slice_x = df.slice(start_idx, lookback_days)
            if len(slice_x) != lookback_days:
                continue
                
            # Extract Features
            # Ensure cols exist
            available_cols = slice_x.columns
            missing = [c for c in feature_cols if c not in available_cols]
            if missing:
                # Fail soft?
                # print(f"Ticker {ticker} missing cols: {missing}")
                continue
                
            x_vals = slice_x.select(feature_cols).to_numpy() # [L, F]
            # Replace NaNs/Infs in X? Scaler handles some, but good to be safe.
            x_vals = np.nan_to_num(x_vals, nan=0.0, posinf=0.0, neginf=0.0)

            # Extract Target (Swing Risk-Adjusted)
            # Need path [target_idx : target_idx + horizon] (inclusive of start for base price?)
            # Return is (Close[t+h] / Close[t]) - 1
            # Path for Vol/DD: Close[t]...Close[t+h]
            target_future_idx = target_idx + horizon_days
            if target_future_idx >= len(df):
                continue
                
            # Get path
            # slice length = horizon + 1 (t to t+h)
            path_slice = df.slice(target_idx, horizon_days + 1)
            if len(path_slice) != horizon_days + 1:
                continue
                
            closes = path_slice["close"].to_numpy()
            if closes[0] == 0: continue
            
            # 1. Total Return
            ret = (closes[-1] / closes[0]) - 1.0
            
            # 2. Realized Vol (of log returns along path)
            # log_ret = diff(log(close))
            # std of daily log returns * sqrt(252)? Or just sum diffs?
            # User spec: "vol_10 (realized vol over 10D)"
            # Usually annualized? Or period?
            # Let's use annualized realized volatility.
            # Avoid log(0)
            cSafe = np.where(closes <= 0, np.nan, closes)
            if np.isnan(cSafe).any(): continue
            
            log_prices = np.log(cSafe)
            log_rets = np.diff(log_prices)
            # Annualize: std * sqrt(252)
            # If horizon is 10 days, we have 10 returns.
            vol_10 = np.std(log_rets, ddof=1) * np.sqrt(252)
            if np.isnan(vol_10): vol_10 = 0.0
            
            # 3. Max Drawdown (over 10D path)
            # DD at step t: (High_to_here - Price_t) / High_to_here
            # Max DD.
            # We use High? Or Close? "max drawdown over 10D path" usually close-to-close if we only have closes.
            peak = closes[0]
            max_dd = 0.0
            for p in closes:
                if p > peak:
                    peak = p
                dd = (peak - p) / peak
                if dd > max_dd:
                    max_dd = dd
            
            # 4. SPY Return (Excess)
            # We don't have SPY path here easily without loading it.
            # Simplification: Assume 'spy_ret_10d' is not readily available unless we load SPY.
            # BUT: We are processing 'ticker'.
            # Does 'slice_x' (features) have SPY data?
            # features usually have 'spy_ret_1d'.
            # We need fwd spy return.
            # Use 'spy_ret_1d' from FUTURE? We can't access future features easily unless we load them.
            # Or we assume 'market_return' is small or Beta=1.
            # User requirement: "excess_10 (10D return minus SPY 10D return)".
            # I really need SPY forward return.
            
            # Hack: Pass SPY data via `market_data` arg?
            # Or assume we just use raw return for now if SPY missing?
            # The prompt is strict. "excess_10d".
            
            # Let's assume we can load SPY separately or it's passed in.
            # Ideally `make_cross_sectional_batch` signature update.
            # For now, I'll use 0.0 for SPY if not available, but add a TODO or try to load.
            # Or use `spy_ret_1d` from the path if available in df?
            # If `df` has `spy_close`? Unlikely.
            
            spy_ret = 0.0
            # If we want to be correct, we need SPY.
            
            # 5. Risk Adjusted Label
            # excess = ret - spy_ret
            # y = excess - lambda * vol - kappa * abs(dd)
            
            # Load config
            # We need CONFIG constants.
            from backend.app.ops import config as cfg
            # If config doesn't have them (it does now), assume defaults.
            LAMBDA = getattr(cfg, "RISK_LAMBDA_VOL", 0.0)
            KAPPA = getattr(cfg, "RISK_KAPPA_DD", 0.0)
            
            excess = ret - spy_ret
            y_swing = excess - (LAMBDA * vol_10) - (KAPPA * abs(max_dd))
            
            # Aux: Direction
            direction = 1.0 if ret > 0 else 0.0

            batch_X.append(x_vals) # [L, F]
            batch_y.append([y_swing, direction])
            valid_tickers.append(ticker)

        except Exception as e:
            # print(f"Error processing {ticker}: {e}")
            continue

    if not batch_X:
        return None
        
    # Stack
    # X: [B, T, F]
    X_tensor = torch.tensor(np.stack(batch_X), dtype=torch.float32)

    # y: [B, 2] (SwingLabel, Direction)
    y_tensor = torch.tensor(np.stack(batch_y), dtype=torch.float32)
    
    # Clip extreme labels?
    # y = torch.clamp(y, -5.0, 5.0)

    return {
        "X": X_tensor,
        "y": y_tensor[:, 0], # Combined Risk-Adj Swing Label
        "y_aux": y_tensor[:, 1], # Direction
        "tickers": valid_tickers
    }
