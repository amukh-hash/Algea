import os
import shutil
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from backend.app.core import config
from backend.app.training.dataset import RankingDataset, ranking_collate_fn
from backend.app.models.rank_transformer import RankTransformer
from backend.app.models.calibration import ScoreCalibrator

def calculate_max_drawdown(cumulative_returns):
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) # log returns
    # or if prices: (price - peak)/peak
    # cum_ret is log sum.
    # dd = cum_ret - peak
    return np.min(drawdown)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Test Data (2024+)
    print(f"Loading Test Data ({config.TEST_SPLIT_DATE}+)...")
    test_dataset = RankingDataset(
        features_path="backend/data/artifacts/features/features_scaled.parquet",
        priors_dir="backend/data/artifacts/priors",
        date_range=(config.TEST_SPLIT_DATE, object()), # Open ended? pd.to_datetime handles None?
        # Dataset expects tuple of strs.
        target_col="log_return_1d" # We backtest on 1d returns
    )
    # Re-init dataset logic for open ended date range if needed
    # ranking_dataset handles it?
    # Actually, dataset logic: 
    # if date_range: start, end = ...
    # pd.to_datetime(None) is NaT.
    # I should pass explicit end date if possible, or update dataset to handle None.
    # Let's pass "2099-12-31" or config.TRAIN_END_DATE.
    
    test_dataset = RankingDataset(
        features_path="backend/data/artifacts/features/features_scaled.parquet",
        priors_dir="backend/data/artifacts/priors",
        date_range=(config.TEST_SPLIT_DATE, config.TRAIN_END_DATE),
        target_col="log_return_1d"
    )

    if len(test_dataset) == 0:
        print("No test data found.")
        return

    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, collate_fn=ranking_collate_fn)
    
    # 2. Load Model & Calibrator
    model = RankTransformer(input_dim=9, pooling="none").to(device)
    model.load_state_dict(torch.load("backend/data/artifacts/models/ranker_v1.pt", map_location=device))
    model.eval()
    
    calib = ScoreCalibrator.load("backend/data/artifacts/models/calibration_v1.joblib")
    
    # 3. Simulate Signal Backtest
    daily_returns = []
    
    print("Running Backtest Simulation...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            feats = batch["features"].to(device)
            priors = batch["priors"].to(device)
            # targets here are log_return_1d (next day return) if dataset configured correctly
            # But RankingDataset defaults to 'target_10d_fwd'.
            # I passed target_col="log_return_1d".
            returns = batch["targets"].cpu().numpy() # [B, L]
            mask = batch["mask"].to(device)
            
            x = torch.cat([feats, priors], dim=-1)
            out = model(x, mask)
            scores = out["score"].squeeze(-1).cpu().numpy() # [B, L]
            
            mask_cpu = mask.cpu().numpy()
            
            # Iterate batch (Days)
            for i in range(len(scores)):
                valid_len = mask_cpu[i].sum()
                if valid_len == 0: continue
                
                day_scores = scores[i, :valid_len]
                day_rets = returns[i, :valid_len]
                
                # Calibrate EV (monotonic transformation, doesn't change rank order)
                # evs = calib.predict(day_scores)
                
                # Top K Selection (Top 10%)
                k = max(1, int(valid_len * 0.10))
                
                # Argpartition/Sort
                # We want large scores
                top_indices = np.argpartition(day_scores, -k)[-k:]
                
                # Avg Return of Top K
                port_ret = np.mean(day_rets[top_indices])
                daily_returns.append(port_ret)
                
    # 4. Metrics
    if not daily_returns:
        print("No returns generated.")
        return
        
    daily_returns = np.array(daily_returns)
    cum_ret = np.cumsum(daily_returns)
    
    total_ret = cum_ret[-1]
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-9) * np.sqrt(252)
    max_dd = calculate_max_drawdown(cum_ret) # Log DD
    hit_rate = np.mean(daily_returns > 0)
    
    print("-" * 30)
    print("GATE METRICS")
    print(f"Total Return: {total_ret:.4f} (Log)")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_dd:.2f}")
    print(f"Hit Rate:     {hit_rate:.2%}")
    print("-" * 30)
    
    # 5. Gate Logic
    # Criteria: Sharpe > 1.2, MaxDD > -0.15 (-15%), HitRate > 0.52
    pass_gate = (sharpe > 1.2) and (max_dd > -0.15) and (hit_rate > 0.52)
    
    if pass_gate:
        print("PASS: Model promoted to production.")
        prod_dir = "backend/data/artifacts/production"
        os.makedirs(prod_dir, exist_ok=True)
        
        shutil.copy("backend/data/artifacts/models/ranker_v1.pt", os.path.join(prod_dir, "ranker_v1.pt"))
        shutil.copy("backend/data/artifacts/models/calibration_v1.joblib", os.path.join(prod_dir, "calibration_v1.joblib"))
        # Copy Scaler too
        shutil.copy("backend/data/artifacts/scalers/selector_v1.joblib", os.path.join(prod_dir, "selector_v1.joblib"))
        
        print(f"Artifacts copied to {prod_dir}.")
    else:
        print("FAIL: Model did not meet criteria.")

if __name__ == "__main__":
    main()
