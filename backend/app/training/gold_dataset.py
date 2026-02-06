import torch
from torch.utils.data import Dataset, Subset
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import polars as pl
import psutil
from collections import OrderedDict

class SimpleLRUCache:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache = OrderedDict()

    def get(self, key):
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

class GoldFuturesWindowDataset(Dataset):
    def __init__(self, files: List[Path], required_cols: Tuple[str, ...],
                 context: int, pred: int, stride_rows: int, 
                 max_windows: int, seed: int, cache_size: int,
                 target_col: str = "close_adj"):
        self.files = files
        self.required_cols = required_cols
        self.target_col = target_col
        self.col_map = {name: i for i, name in enumerate(required_cols)}
        
        self.context = context
        self.pred = pred
        self.stride_rows = stride_rows
        self.max_windows = max_windows
        self.rng = np.random.RandomState(seed)
        
        self.cache = SimpleLRUCache(cache_size)
        self.index = [] # List[(file_idx, start_row, timestamp_val)]
        
        print(f"[Dataset] Indexing {len(files)} files...")
        
        for fi, fp in enumerate(self.files):
            try:
                # Use scan to get height without loading data
                n = pl.scan_parquet(fp).select(pl.len()).collect().item()
                
                max_start = n - (context + pred)
                if max_start <= 0: continue
                
                starts = list(range(0, max_start, stride_rows))
                
                # Filter caps
                if len(starts) > max_windows:
                     starts = self.rng.choice(starts, size=max_windows, replace=False).tolist()
                
                for s in starts:
                    # Use row index as t_val for stable sorting if timestamp missing
                    self.index.append((fi, int(s), int(s)))
                    
            except Exception as e:
                print(f"ERR indexing {fp}: {e}")
        
        # Sort by time for temporal split
        try:
             self.index.sort(key=lambda x: x[2])
        except:
             print("WARN: Could not sort index by timestamp (mixed types?). Shuffling instead.")
             self.rng.shuffle(self.index)

    def split_validation(self, split_pct: float) -> Tuple[Subset, Subset]:
        total = len(self.index)
        val_size = int(total * split_pct)
        train_size = total - val_size
        
        # Temporal split: Train first, Val last
        train_idx = list(range(train_size))
        val_idx = list(range(train_size, total))
        
        return (
            Subset(self, train_idx),
            Subset(self, val_idx)
        )

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        fi, s, _ = self.index[idx]
        fp = self.files[fi]
        
        # Cache access
        arr = self.cache.get(fi)
        if arr is None:
            # Flexible read: Only read columns that exist
            # Note: Using scan logic inside getitem might be slow if overhead is high
            # But pl.scan_parquet is generally light? 
            # Actually, scan_parquet -> collect involves file open.
            # Ideally we have file handles or something, but parquet is file-per-op usually.
            
            # Using read_parquet might be simpler if we read all cols anyway?
            # But we only want 'actual_cols'.
            
            # Optimization: Try standard read if schema known?
            # Sticking to original logic for correctness, removed gc.collect()
            
            try:
                schema = pl.scan_parquet(fp).schema
                actual_cols = [c for c in self.required_cols if c in schema]
                df = pl.scan_parquet(fp).select(actual_cols).collect()
                arr = df.to_numpy().astype(np.float32)
                # del df # ref counting handles it
                self.cache.put(fi, arr)
            except Exception as e:
                print(f"Error reading {fp}: {e}")
                # Return dummy
                return {
                    "x_float": np.zeros((self.context, len(self.required_cols)), dtype=np.float32),
                    "y_float": np.zeros((self.pred, len(self.required_cols)), dtype=np.float32)
                }

        # Slicing
        ts_idx = self.col_map.get("timestamp")
        date_idx = self.col_map.get("date")
        start_row = s
        end_row = s + self.context + self.pred
        
        # Safety clip
        if end_row > len(arr):
             end_row = len(arr)
             
        subset = arr[start_row:end_row]
        
        # Exclude non-feature columns (date, timestamp, etc.)
        exclude_indices = set()
        if ts_idx is not None:
            exclude_indices.add(ts_idx)
        if date_idx is not None:
            exclude_indices.add(date_idx)
            
        if exclude_indices:
            feat_indices = [i for i in range(arr.shape[1]) if i not in exclude_indices]
            feats = subset[:, feat_indices].astype(np.float32)
            # We need to track where the target usage is in the NEW 'feats' array
            # Mapping old index to new index
            # old_idx -> new_idx map
            # This is complex without knowing exact indices.
            # Simpler: find index of target_col in required_cols, subtract excluded count before it?
            # Or just re-lookup:
            filtered_cols = [self.required_cols[i] for i in feat_indices]
            try:
                target_feat_idx = filtered_cols.index(self.target_col)
            except ValueError:
                # Target col missing from features (maybe explicit exclusion?)
                target_feat_idx = -1
        else:
            feats = subset.astype(np.float32)
            try:
                 target_feat_idx = self.required_cols.index(self.target_col)
            except ValueError:
                 target_feat_idx = -1

        # Pad if short (rare, due to clip)
        if len(feats) < (self.context + self.pred):
             pad_len = (self.context + self.pred) - len(feats)
             feats = np.pad(feats, ((0, pad_len), (0, 0)))

        # Phase 2: Apply Log-Return Transform to Target Column
        # r_{t+k} = log(price_{t+k} / price_{t})
        # We apply this to whole window relative to the 'split point' t (end of context)
        # Identify reference price: last element of CONTEXT
        # feats is [Context + Pred]
        # Context ends at index self.context - 1
        
        if target_feat_idx >= 0:
            target_series = feats[:, target_feat_idx]
            ref_idx = self.context - 1
            # Ensure ref_idx is valid
            if ref_idx < len(target_series):
                ref_val = target_series[ref_idx]
                # Avoid div by zero or log(0)
                if ref_val > 1e-6:
                     # Transform: log(p / p_ref) = log(p) - log(p_ref)
                     # Vectorized
                     # Use numpy log
                     feats[:, target_feat_idx] = np.log(target_series / ref_val)
                else:
                     # Fallback for bad data: zero out or keep raw?
                     # Zeroing out implies 0 return (flat)
                     feats[:, target_feat_idx] = 0.0

        return {
            "x_float": feats[:self.context],
            "y_float": feats[self.context:]
        }
