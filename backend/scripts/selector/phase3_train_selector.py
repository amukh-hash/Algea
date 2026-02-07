"""
Script to train the Rank-Transformer Selector.
This replaces the old student distillation phase for Swing.
"""

import sys
import os
import argparse
import polars as pl
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime, timedelta
from pathlib import Path
from typing import List
import joblib

# Adjust path
sys.path.append(os.getcwd())

from backend.app.models.rank_transformer import RankTransformer
from backend.app.models.rank_losses import listwise_softmax_loss, pairwise_margin_loss
from backend.app.models.selector_scaler import SelectorFeatureScaler
from backend.app.models.calibration import ScoreCalibrator
from backend.app.data.windows import make_cross_sectional_batch
from backend.app.models.schema import FeatureContract
from backend.app.core import config
from backend.app.ops import run_recorder

# Configuration
VERSION = "v1"
CHECKPOINT_DIR = f"backend/data/checkpoints/selector/{VERSION}"
LOG_DIR = f"backend/data/logs/selector/{VERSION}"

class CrossSectionalDataset(Dataset):
    def __init__(self, dates: List[datetime.date], universe: List[str], data_dir: str, lookback: int, horizon: int, feature_cols: List[str]):
        self.dates = dates
        self.universe = universe
        self.data_dir = data_dir
        self.lookback = lookback
        self.horizon = horizon
        self.feature_cols = feature_cols

    def __len__(self):
        return len(self.dates)

    def __getitem__(self, idx):
        date = self.dates[idx]
        batch = make_cross_sectional_batch(
            target_date=date,
            universe=self.universe,
            data_dir=self.data_dir,
            breadth_path="backend/data/breadth.parquet",
            lookback_days=self.lookback,
            horizon_days=self.horizon,
            feature_cols=self.feature_cols
        )
        if batch is None:
            # Handle empty batch? Return empty tensors or skip?
            # Ideally filter dates beforehand.
            return {
                "X": torch.empty(0),
                "y": torch.empty(0),
                "valid": False
            }

        return {
            "X": batch["X"], # [N, T, F]
            "y": batch["y"], # [N]
            "y_aux": batch["y_aux"], # [N]
            "valid": True,
            "date": str(date)
        }

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=1) # One date per batch step (simpler for variable N)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_id = run_recorder.init_run(
        pipeline_type="selector",
        trigger="manual",
        config={"epochs": args.epochs, "batch_size": args.batch_size, "lr": args.lr, "device": args.device},
        data_versions={"gold": "unknown", "silver": "unknown", "macro": "unknown", "universe": "unknown"},
        tags=["selector", "training"],
    )
    run_dir = run_recorder.run_paths.get_run_dir(run_id)
    checkpoint_dir = run_dir / "checkpoints"
    log_dir = run_dir / "outputs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    run_recorder.set_status(run_id, "RUNNING", stage="train", step="init")

    priors_dir = Path(config.PRIORS_DIR) / "chronos2" / "v1"
    if not priors_dir.exists() or not list(priors_dir.glob("*.parquet")):
        run_recorder.set_status(
            run_id,
            "FAILED",
            stage="train",
            step="priors_check",
            error={
                "type": "MissingPriors",
                "message": "No priors artifact found in backend/data/priors/chronos2/v1.",
                "traceback": "Run nightly_build_priors.",
            },
        )
        run_recorder.finalize_run(run_id, "FAILED")
        return

    # 1. Setup Data
    # Assume we have a list of dates
    # start_date = datetime(2020, 1, 1).date()
    # end_date = datetime(2023, 12, 31).date()
    dates = [datetime(2023, 1, 1).date() + timedelta(days=x) for x in range(300)] # Mock dates

    # Filter dates to trading days?
    # Assume make_cross_sectional_batch handles non-trading days by returning None.

    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY"] # Mock
    feature_cols = FeatureContract.CORE_FEATURES + FeatureContract.MARKET_FEATURES + FeatureContract.PRIOR_FEATURES

    # Fit Scaler First?
    # Ideally fit on training split only.
    # Load sample batch to fit scaler.
    print("Fitting scaler...")
    scaler = SelectorFeatureScaler(version=VERSION, feature_names=feature_cols)

    # Collect some data for fitting
    sample_X = []
    for d in dates[:20]:
        b = make_cross_sectional_batch(d, universe, "backend/data/features", "backend/data/breadth.parquet", 60, 10, feature_cols)
        if b:
            sample_X.append(b["X"].numpy())

    if sample_X:
        # Stack: [Total, T, F]
        all_X = np.concatenate(sample_X, axis=0)
        scaler.fit(all_X)
        scaler_path = checkpoint_dir / "scaler.joblib"
        scaler.save(str(scaler_path))
        run_recorder.register_artifact(
            run_id,
            name="selector_scaler",
            type="model_checkpoint",
            path=str(scaler_path),
            tags=["scaler"],
        )
    else:
        print("Warning: No data found to fit scaler. Using unfitted (no-op or error).")

    # Dataset
    train_ds = CrossSectionalDataset(dates[:250], universe, "backend/data/features", 60, 10, feature_cols)
    val_ds = CrossSectionalDataset(dates[250:], universe, "backend/data/features", 60, 10, feature_cols)

    # DataLoader: batch_size=1 means one date at a time.
    # Collate fn: just return the single item dict since shapes vary per date.
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, collate_fn=lambda x: x[0])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])

    # Model
    model = RankTransformer(d_input=len(feature_cols)).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    # Loop
    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        steps = 0

        for batch in train_loader:
            if not batch["valid"]:
                continue

            X = batch["X"][:, -1, :].to(args.device) # [N, F]
            y = batch["y"].to(args.device) # [N]

            # Scale
            X = scaler.transform(X).unsqueeze(0)

            optimizer.zero_grad()

            out = model(X)
            scores = out["score"].squeeze(0).squeeze(-1) # [N]

            # Loss: Listwise + Pairwise
            loss_list = listwise_softmax_loss(scores, y)
            loss_pair = pairwise_margin_loss(scores, y)

            loss = loss_list + 0.5 * loss_pair

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            steps += 1

        avg_train_loss = train_loss / max(steps, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_steps = 0
        all_scores = []
        all_targets = []

        with torch.no_grad():
            for batch in val_loader:
                if not batch["valid"]:
                    continue

                X = batch["X"][:, -1, :].to(args.device)
                y = batch["y"].to(args.device)

                X = scaler.transform(X).unsqueeze(0)
                out = model(X)
                scores = out["score"].squeeze(0).squeeze(-1)

                loss = listwise_softmax_loss(scores, y) + 0.5 * pairwise_margin_loss(scores, y)
                val_loss += loss.item()
                val_steps += 1

                all_scores.append(scores.cpu().numpy())
                all_targets.append(y.cpu().numpy())

        avg_val_loss = val_loss / max(val_steps, 1)
        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Val Loss {avg_val_loss:.4f}")
        run_recorder.emit_metric(
            run_id,
            step=epoch + 1,
            epoch=float(epoch + 1),
            metrics={"train_loss": avg_train_loss, "val_loss": avg_val_loss},
        )
        run_recorder.set_status(run_id, "RUNNING", stage="train", step=f"epoch_{epoch + 1}")

        # Checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = checkpoint_dir / "best.pt"
            torch.save(model.state_dict(), best_path)
            run_recorder.register_checkpoint(
                run_id,
                checkpoint_path=str(best_path),
                step=epoch + 1,
                epoch=float(epoch + 1),
                component="selector",
                meta={"val_loss": avg_val_loss},
            )

            # Fit Calibrator on Val set
            if all_scores:
                calibrator = ScoreCalibrator(version=VERSION)
                flat_scores = np.concatenate(all_scores).ravel()
                flat_targets = np.concatenate(all_targets).ravel()
                calibrator.fit(flat_scores, flat_targets)
                calibrator_path = checkpoint_dir / "calibration.joblib"
                calibrator.save(str(calibrator_path))
                run_recorder.register_artifact(
                    run_id,
                    name="selector_calibration",
                    type="model_checkpoint",
                    path=str(calibrator_path),
                    tags=["calibration"],
                )

    print("Training Complete.")
    run_recorder.finalize_run(run_id, "PASSED")

if __name__ == "__main__":
    train()
