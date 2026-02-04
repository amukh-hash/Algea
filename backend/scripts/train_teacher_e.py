import os
import sys
import torch
import polars as pl
import argparse
from torch.utils.data import DataLoader
from datetime import datetime

# Ensure backend in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from backend.app.data import windows, manifest
from backend.app.preprocessing.preproc import Preprocessor
from backend.app.models.baseline import BaselineMLP
from backend.app.models import model_io
from backend.app.models.signal_types import ModelMetadata

def train(args):
    # 1. Load Data
    # For baseline, load one ticker or concat multiple?
    # Let's load AAPL as representative.
    ticker = "AAPL" # Hardcoded for test
    mf_path = os.path.join(args.data_dir, f"marketframe_{ticker}_1m.parquet")
    if not os.path.exists(mf_path):
        print(f"Data not found: {mf_path}")
        return

    df = pl.read_parquet(mf_path)
    
    # 2. Load Preproc
    preproc = Preprocessor.load(args.preproc_path)
    
    # 3. Transform Data
    # We transform the whole DF first for speed? 
    # Yes, SwingWindowDataset takes transformed features usually, or transforms on fly?
    # My SwingWindowDataset implementation takes `input_cols` from `df`.
    # It assumes `df` has those columns.
    # So we must transform first.
    print("Preprocessing data...")
    df_trans = preproc.transform(df)
    
    # Add back 'close' for target calculation if missing
    if "close" not in df_trans.columns:
        df_trans = df_trans.with_columns(df["close"])

    # 4. Create Dataset
    cols = ["log_ret", "volume_norm", "ad_line_norm", "bpi_norm"]
    lookback = 128
    ds = windows.SwingWindowDataset(df_trans, cols, lookback=lookback, stride=60) # Stride 60 mins
    
    if len(ds) == 0:
        print("Dataset empty. Check lookback/data size.")
        return
        
    loader = DataLoader(ds, batch_size=32, shuffle=True)
    
    # 5. Init Model
    model = BaselineMLP(input_dim=len(cols), lookback=lookback)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss() # Dummy loss for quantiles
    
    # 6. Train Loop
    print("Training...")
    model.train()
    for epoch in range(1):
        total_loss = 0
        batches = 0
        for x, y in loader:
            # y is dict {"1D": ..., "3D": ...}
            # Target: we want to predict quantiles.
            # For baseline, let's just predict the mean return for now -> MSE
            # or try to predict 3 values?
            # Model outputs (B, H, 3).
            # We treat y as ground truth scalar. 
            # Pinball loss is better for quantiles.
            # For "functional baseline", MSE against the mean output is fine.
            # We optimize the middle quantile (index 1) to match y.
            
            y_1d = y["1D"].float().unsqueeze(-1) # (B, 1)
            y_3d = y["3D"].float().unsqueeze(-1) # (B, 1)
            
            optimizer.zero_grad()
            out = model(x) # (B, 2, 3)
            
            # Predict mean (index 1)
            pred_1d = out[:, 0, 1].unsqueeze(-1)
            pred_3d = out[:, 1, 1].unsqueeze(-1)
            
            loss = criterion(pred_1d, y_1d) + criterion(pred_3d, y_3d)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            batches += 1
            if batches > 10: break # Short run for smoke test
            
        print(f"Epoch {epoch} Loss: {total_loss/batches}")
        
    # 7. Save
    meta = ModelMetadata(
        model_version="v1_baseline",
        preproc_id=preproc.version_hash,
        training_start="?",
        training_end="?"
    )
    
    model_io.save_model(model.state_dict(), meta, args.output_path)
    print(f"Saved model to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="backend/data/marketframe")
    parser.add_argument("--preproc_path", default="backend/models/preproc/preproc_v1.json")
    parser.add_argument("--output_path", default="backend/models/teacher_e/teacher_v1.pt")
    args = parser.parse_args()
    
    train(args)
