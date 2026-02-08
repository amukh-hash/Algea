import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional

class SelectorDataset(Dataset):
    """
    SelectorDataset V2
    Loads SelectorFeatureFrame V2 (Parquet) for Two-Head Rank Selector.
    
    Item: (Date, X[N, F], y_rank[N], y_trade[N], w[N], tiers[N])
    """
    def __init__(self, 
                 features_path: str, # selector_features.parquet
                 date_range: Tuple[str, str] = None,
                 feature_cols: List[str] = ["x_lr1", "x_lr5", "x_lr20", "x_vol", "x_relvol"]):
        
        self.features_df = pd.read_parquet(features_path)
        
        # Ensure date type
        if not pd.api.types.is_datetime64_any_dtype(self.features_df['date']):
            self.features_df['date'] = pd.to_datetime(self.features_df['date'])
            
        # Filter by date range
        if date_range:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            self.features_df = self.features_df[
                (self.features_df['date'] >= start) & (self.features_df['date'] <= end)
            ].copy()
            
        # Ensure weights > 0 (Safety)
        # UniverseFrame should have handled this, but good to check.
        # If weight is missing, default to 1.0?
        if 'weight' not in self.features_df.columns:
            self.features_df['weight'] = 1.0
            
        # Filter out zero weights just in case
        self.features_df = self.features_df[self.features_df['weight'] > 0].copy()
            
        # Index unique dates
        self.dates = sorted(self.features_df['date'].unique())
        self.feature_cols = feature_cols
        
        # Tiers mapping (Optional one-hot?)
        # For now return integer or raw string?
        # Let's return raw string array for analysis or one-hot/integer for embedding?
        # Model might use tier embedding.
        self.tier_map = {'A': 0, 'B': 1, 'C': 2} 
        
    def __len__(self):
        return len(self.dates)
        
    def __getitem__(self, idx):
        date = self.dates[idx]
        date_str = date.strftime("%Y-%m-%d")
        
        # Slice for the day
        day_df = self.features_df[self.features_df['date'] == date]
        
        # 1. Features X [N, 5]
        X = day_df[self.feature_cols].values.astype(np.float32)
        
        # 2. Targets
        # y_rank [N]
        # Handle missings? y_rank might be NaN if at end of history.
        # Training loop should handle NaNs or we fill with 0 and mask?
        # Usually drop rows with NaN targets during training preparation?
        # Or here we fillna(0) and let loss function mask it?
        # Let's fill 0.
        y_rank = day_df['y_rank'].fillna(0.0).values.astype(np.float32)
        
        # y_trade [N] (0 or 1)
        y_trade = day_df['y_trade'].fillna(0).values.astype(np.float32)
        
        # 3. Weights [N]
        w = day_df['weight'].values.astype(np.float32)
        
        # 4. Tiers
        # Map to int
        tiers = day_df['tier'].map(self.tier_map).fillna(-1).values.astype(np.int64)
        
        return {
            "date": date_str,
            "tickers": day_df['ticker'].values,
            "X": torch.tensor(X),
            "y_rank": torch.tensor(y_rank),
            "y_trade": torch.tensor(y_trade),
            "w": torch.tensor(w),
            "tiers": torch.tensor(tiers)
        }

def selector_collate_fn(batch):
    """
    Pads batch to max sequence length N_max in batch.
    """
    # max N in batch
    max_len = max([b["X"].shape[0] for b in batch])
    B = len(batch)
    F = batch[0]["X"].shape[1]
    
    # Init Padded Tensors
    X_pad = torch.zeros(B, max_len, F)
    y_rank_pad = torch.zeros(B, max_len)
    y_trade_pad = torch.zeros(B, max_len)
    w_pad = torch.zeros(B, max_len) # Weights 0 for padding
    tiers_pad = torch.full((B, max_len), -1, dtype=torch.long)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    
    dates = []
    tickers_list = []
    
    for i, b in enumerate(batch):
        L = b["X"].shape[0]
        X_pad[i, :L, :] = b["X"]
        y_rank_pad[i, :L] = b["y_rank"]
        y_trade_pad[i, :L] = b["y_trade"]
        w_pad[i, :L] = b["w"]
        tiers_pad[i, :L] = b["tiers"]
        mask[i, :L] = True
        
        dates.append(b["date"])
        tickers_list.append(b["tickers"])
        
    return {
        "X": X_pad,             # [B, N_max, F]
        "y_rank": y_rank_pad,   # [B, N_max]
        "y_trade": y_trade_pad, # [B, N_max]
        "w": w_pad,             # [B, N_max]
        "tiers": tiers_pad,     # [B, N_max]
        "mask": mask,           # [B, N_max]
        "dates": dates,
        "tickers": tickers_list
    }
