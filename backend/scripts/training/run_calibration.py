import torch
import os
import joblib
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from backend.app.core import config
from backend.app.training.dataset import RankingDataset, ranking_collate_fn
from backend.app.models.rank_transformer import RankTransformer
from backend.app.models.calibration import ScoreCalibrator

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Val Data (2023)
    val_dataset = RankingDataset(
        features_path="backend/data/artifacts/features/features_scaled.parquet",
        priors_dir="backend/data/artifacts/priors",
        date_range=(config.TRAIN_SPLIT_DATE, config.TEST_SPLIT_DATE), # 2023-2024
        target_col="target_10d_fwd" # Ensure targets exist
    )
    
    if len(val_dataset) == 0:
        print("No validation data found. Skipping calibration.")
        return
        
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=ranking_collate_fn)
    
    # 2. Load Model
    model = RankTransformer(input_dim=9).to(device)
    model.load_state_dict(torch.load("backend/data/artifacts/models/ranker_v1.pt", map_location=device))
    model.eval()
    
    all_scores = []
    all_targets = []
    
    print("Generating scores for calibration...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            feats = batch["features"].to(device)
            priors = batch["priors"].to(device)
            targets = batch["targets"].cpu().numpy() # [B, L]
            mask = batch["mask"].to(device) # [B, L]
            
            x = torch.cat([feats, priors], dim=-1)
            out = model(x, mask)
            scores = out["score"].squeeze(-1) # [B, L]
            
            # Mask out padding scores
            mask_cpu = mask.cpu().numpy()
            scores_cpu = scores.cpu().numpy()
            
            for i in range(len(targets)):
                # Filter by length
                valid_len = mask_cpu[i].sum()
                s = scores_cpu[i, :valid_len]
                t = targets[i, :valid_len]
                
                all_scores.append(s)
                all_targets.append(t)
                
    # Concatenate
    flat_scores = np.concatenate(all_scores)
    flat_targets = np.concatenate(all_targets)
    
    # 3. Fit Calibrator
    print(f"Fitting Calibrator on {len(flat_scores)} points...")
    calib = ScoreCalibrator(version="v1")
    calib.fit(flat_scores, flat_targets)
    
    # 4. Save
    out_path = "backend/data/artifacts/models/calibration_v1.joblib"
    calib.save(out_path)
    print(f"Saved Calibrator to {out_path}")

if __name__ == "__main__":
    main()
