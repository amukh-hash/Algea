import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os
import glob
from typing import List, Dict, Tuple

class RankingDataset(Dataset):
    """
    Loads daily cross-sections for Listwise Ranking.
    Item: (Date, Features[N, F], Priors[N, P], Targets[N])
    """
    def __init__(self, 
                 features_path: str, # features_scaled.parquet
                 priors_dir: str, # backend/data/artifacts/priors/
                 target_col: str = "target_10d_fwd",
                 date_range: Tuple[str, str] = None):
        
        self.features_df = pd.read_parquet(features_path)
        self.features_df['date'] = pd.to_datetime(self.features_df['date'])
        
        if date_range:
            start, end = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            self.features_df = self.features_df[(self.features_df['date'] >= start) & (self.features_df['date'] <= end)]
            
        # Index unique dates
        self.dates = sorted(self.features_df['date'].unique())
        self.priors_dir = priors_dir
        
        # Feature columns (exclude meta)
        # We assume scaler transformed correctly and we just pick float cols
        # Or hardcode?
        exclude = ['date', 'ticker', target_col, 'target_10d_fwd_bin']
        self.feat_cols = [c for c in self.features_df.columns if c not in exclude]
        
    def __len__(self):
        return len(self.dates)
        
    def __getitem__(self, idx):
        date = self.dates[idx]
        date_str = date.strftime("%Y-%m-%d")
        
        # 1. Get Day Slice
        day_df = self.features_df[self.features_df['date'] == date].copy()
        
        # 2. Get Priors
        # priors_{date}.parquet
        priors_path = os.path.join(self.priors_dir, f"priors_{date_str}.parquet")
        
        if os.path.exists(priors_path):
            priors_df = pd.read_parquet(priors_path)
            # Merge
            day_df = day_df.merge(priors_df, on="ticker", how="left")
        else:
            # Missing priors? Pad with 0 or skip?
            # Model needs priors. 
            # We add dummy columns if missing?
            # Better to error or handle. 
            pass # Fillna later
            
        # 3. Align Columns
        # Features
        X_feats = day_df[self.feat_cols].values.astype(np.float32)
        
        # Priors (drift, vol, downside, conf)
        prior_cols = ["prior_drift_20d", "prior_vol_20d", "prior_downside_q10", "prior_trend_conf"]
        # Ensure cols exist
        for c in prior_cols:
            if c not in day_df.columns:
                day_df[c] = 0.0
                
        X_priors = day_df[prior_cols].fillna(0.0).values.astype(np.float32)
        
        # Targets
        if "target_10d_fwd" in day_df.columns:
            y = day_df["target_10d_fwd"].fillna(0.0).values.astype(np.float32)
        else:
            y = np.zeros(len(day_df), dtype=np.float32)
            
        # Pad to fixed size? Or use Collate fn?
        # Ranking models often handle variable list size via mask.
        # Here we return variable size tensors and let collate handle padding.
        
        return {
            "features": torch.tensor(X_feats),
            "priors": torch.tensor(X_priors),
            "targets": torch.tensor(y),
            "date": date_str,
            "tickers": day_df['ticker'].values
        }

def ranking_collate_fn(batch):
    """
    Pads batch to max sequence length in batch.
    """
    # batch is list of dicts
    max_len = max([b["features"].shape[0] for b in batch])
    
    # Init Padded Tensors
    B = len(batch)
    F_feats = batch[0]["features"].shape[1]
    F_priors = batch[0]["priors"].shape[1]
    
    padded_feats = torch.zeros(B, max_len, F_feats)
    padded_priors = torch.zeros(B, max_len, F_priors)
    padded_targets = torch.zeros(B, max_len)
    mask = torch.zeros(B, max_len, dtype=torch.bool)
    
    for i, b in enumerate(batch):
        L = b["features"].shape[0]
        padded_feats[i, :L, :] = b["features"]
        padded_priors[i, :L, :] = b["priors"]
        padded_targets[i, :L] = b["targets"]
        mask[i, :L] = True
        
    return {
        "features": padded_feats,
        "priors": padded_priors,
        "targets": padded_targets,
        "mask": mask
    }
