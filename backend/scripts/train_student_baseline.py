import os
import sys
import torch
import polars as pl
import argparse
from torch.utils.data import DataLoader

# Ensure backend in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from backend.app.data import windows
from backend.app.preprocessing.preproc import Preprocessor
from backend.app.models.baseline import BaselineMLP
from backend.app.models import model_io
from backend.app.models.signal_types import ModelMetadata

def train(args):
    ticker = "AAPL"
    mf_path = os.path.join(args.data_dir, f"marketframe_{ticker}_1m.parquet")
    if not os.path.exists(mf_path):
        print(f"Data not found: {mf_path}")
        return

    df = pl.read_parquet(mf_path)
    preproc = Preprocessor.load(args.preproc_path)
    print("Preprocessing data...")
    df_trans = preproc.transform(df)

    if "close" not in df_trans.columns:
        df_trans = df_trans.with_columns(df["close"])

    cols = ["log_ret", "volume_norm", "ad_line_norm", "bpi_norm"]
    lookback = 128
    ds = windows.SwingWindowDataset(df_trans, cols, lookback=lookback, stride=60)

    if len(ds) == 0:
        print("Dataset empty.")
        return

    loader = DataLoader(ds, batch_size=32, shuffle=True)

    model = BaselineMLP(input_dim=len(cols), lookback=lookback)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()

    print("Training Student Baseline...")
    model.train()
    for epoch in range(1):
        total_loss = 0
        batches = 0
        for x, y in loader:
            y_1d = y["1D"].float().unsqueeze(-1)
            y_3d = y["3D"].float().unsqueeze(-1)

            optimizer.zero_grad()
            out = model(x)
            pred_1d = out[:, 0, 1].unsqueeze(-1)
            pred_3d = out[:, 1, 1].unsqueeze(-1)

            loss = criterion(pred_1d, y_1d) + criterion(pred_3d, y_3d)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batches += 1
            if batches > 10: break

        print(f"Epoch {epoch} Loss: {total_loss/batches}")

    meta = ModelMetadata(
        model_version="v1_student_baseline",
        preproc_id=preproc.version_hash,
        training_start="?",
        training_end="?"
    )

    model_io.save_model(model.state_dict(), meta, args.output_path)
    print(f"Saved student model to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="backend/data/marketframe")
    parser.add_argument("--preproc_path", default="backend/models/preproc/preproc_v1.json")
    parser.add_argument("--output_path", default="backend/models/student/student_v1.pt")
    args = parser.parse_args()

    train(args)
