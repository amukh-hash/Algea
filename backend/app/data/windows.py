import torch
from torch.utils.data import Dataset
import polars as pl
import numpy as np
import pandas as pd
from typing import List, Optional
from backend.app.data import calendar

class SwingWindowDataset(Dataset):
    def __init__(self, df: pl.DataFrame, input_cols: List[str], lookback: int, stride: int = 1, horizons: List[str] = ["1D", "3D"]):
        """
        df: Polars DataFrame (single ticker, sorted by time).
        input_cols: List of column names to use as features.
        lookback: Number of time steps (minutes) for the encoder.
        stride: Stride for sampling start points.
        horizons: List of horizons to compute targets for. Supported: "1D", "3D".
        """
        self.lookback = lookback
        self.input_cols = input_cols
        
        # Convert to pandas/numpy for easier indexing for now? 
        # Polars is fast but random access by index is also fast if we extract to numpy.
        self.timestamps = df["timestamp"].to_numpy() # array of numpy.datetime64
        self.closes = df["close"].to_numpy().astype(np.float32)
        
        # Extract features matrix
        # Shape: (T, F)
        self.features = df.select(input_cols).to_numpy().astype(np.float32)
        
        self.stride = stride
        self.horizons = horizons
        
        # Valid start indices
        # We need index i such that:
        # 1. i >= lookback - 1 (so we have 'lookback' steps ending at i)
        # 2. Targets exist.
        
        # Let's precompute targets mapping.
        # mapping: index -> { "1D": target_val, "3D": target_val }
        # Or better: parallel arrays for targets.
        
        self.targets = {h: np.full(len(self.timestamps), np.nan, dtype=np.float32) for h in horizons}
        self.valid_indices = []
        
        self._compute_targets()
        
    def _compute_targets(self):
        # We need a map from timestamp to index and price
        # ts_map = {ts: (idx, price) ...} 
        # Too slow for millions of rows?
        # Use searchsorted on timestamps.
        
        # Optimize: 
        # Calculate target timestamps for ALL indices we might use.
        # But `get_next_session_close` is expensive to call 1M times.
        # However, for 1m data, many minutes map to the same session close.
        # We can cache the session close for each unique DAY or SESSION.
        # Or just iterate.
        
        # For simplicity and correctness in Phase 1, we iterate but maybe skip based on stride.
        # But we need valid_indices for __len__.
        
        # Let's try to be efficient.
        # Convert all timestamps to pandas for calendar util?
        # calendar util expects pandas Timestamp or datetime.
        
        # Create a helper to map "minute" -> "next_session_close_ts"
        # Since 'next_session_close' is monotonic, maybe we can optimize?
        # Actually, for 1D, for all minutes in a session, the target is that session's close (or next if after).
        # We can just identify the session for each timestamp.
        
        cal = calendar.get_calendar()
        
        # Convert timestamps to UTC pandas index
        pd_ts = pd.to_datetime(self.timestamps).tz_localize("UTC") # Assuming inputs are UTC naive from Polars/Parquet? 
        # Wait, Polars read_parquet might preserve TZ.
        # Check first element.
        if len(self.timestamps) > 0:
            # np.datetime64 usually naive unless localized.
            pass

        # We need to populate self.targets and self.valid_indices
        # We'll iterate with stride.
        
        # Create a lookup for Close prices by Timestamp
        # Exact match required? 
        # Yes, target is the Close price at the session close time.
        # But the session close time (e.g. 16:00:00) might exist in our data.
        # If it doesn't (e.g. data ends 15:59:00), do we use 15:59?
        # Standard: use last available bar <= close_time?
        # Or assume our MarketFrame has the close bar.
        # We'll use searchsorted to find nearest <=.
        
        # Pre-calculate session closes for the range
        # This is optimization. For now, let's just do the loop for stride indices.
        # It might take a few seconds for a year of data.
        
        # We only care about indices where i >= lookback-1
        start_idx = self.lookback - 1
        
        # We'll store valid indices
        indices = []
        
        # To speed up:
        # We can vector-calculate session closes if we map each TS to session.
        # But simpler:
        
        for i in range(start_idx, len(self.timestamps), self.stride):
            curr_ts = pd_ts[i]
            curr_close = self.closes[i]
            
            valid = True
            row_targets = {}
            
            for h in self.horizons:
                n = 1 if h == "1D" else 3
                try:
                    target_ts = calendar.get_next_session_close(curr_ts, n_sessions=n)
                except:
                    valid = False
                    break
                
                # Find target_ts in self.timestamps
                # searchsorted returns index where it should be inserted to maintain order.
                # side='right' -> index after last occurrence.
                # if exact match, ts[idx-1] == target_ts?
                
                # We want exact match or closest previous? 
                # Target is "Session Close". Ideally exact.
                # If exact missing, allow small tolerance?
                
                # Using searchsorted on numpy array of datetimes
                # self.timestamps is np.datetime64[ns]
                # target_ts is pd.Timestamp -> convert to np.datetime64
                target_ts_np = target_ts.to_datetime64()
                
                idx = np.searchsorted(self.timestamps, target_ts_np, side='left')
                
                # Check if exact match
                if idx < len(self.timestamps) and self.timestamps[idx] == target_ts_np:
                    target_price = self.closes[idx]
                else:
                    # Not found exactly.
                    # Try idx-1 (maybe 15:59?)
                    # Tolerance: 5 minutes?
                    # If we accept 15:59 as Close.
                    if idx > 0:
                        prev = self.timestamps[idx-1]
                        diff = target_ts_np - prev
                        # 5 mins in ns = 5 * 60 * 1e9 = 300e9
                        if diff <= np.timedelta64(5, 'm'):
                             target_price = self.closes[idx-1]
                        else:
                            valid = False
                            break
                    else:
                        valid = False
                        break
                
                # Calculate Log Return
                # ln(Pt / P0)
                if curr_close <= 1e-9 or target_price <= 1e-9:
                     valid = False
                     break
                     
                ret = np.log(target_price / curr_close)
                row_targets[h] = ret
            
            if valid:
                for h in self.horizons:
                    self.targets[h][i] = row_targets[h]
                indices.append(i)
                
        self.valid_indices = indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        # idx is map to real index
        real_idx = self.valid_indices[idx]
        
        # Get window
        # [real_idx - lookback + 1 : real_idx + 1]
        start = real_idx - self.lookback + 1
        end = real_idx + 1
        
        x = self.features[start:end] # Shape (Lookback, F)
        
        y = {h: self.targets[h][real_idx] for h in self.horizons}
        
        return torch.from_numpy(x), y
