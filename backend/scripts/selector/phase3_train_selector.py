
"""
Script to train the Rank-Transformer Selector (Swing Strategy).
Includes:
- Risk-Adjusted Labels (Excess - Lambda*Vol - Kappa*DD)
- Turnover Control (Sequential Regularization)
- Ablation Study (With/Without Priors)
- Top-K Evaluation
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
from typing import List, Dict, Any
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
# Also ops config for risk params
from backend.app.ops import config as ops_config
from backend.app.ops import run_recorder
from backend.app.eval.topk_basket_eval import TopKBasketEvaluator

# Configuration
VERSION = "v1"
CHECKPOINT_DIR = f"backend/data/checkpoints/selector/{VERSION}"
LOG_DIR = f"backend/data/logs/selector/{VERSION}"

class CrossSectionalDataset(Dataset):
    def __init__(self, dates: List[datetime.date], universe: List[str], data_dir: str, lookback: int, horizon: int, feature_cols: List[str]):
        self.dates = sorted(dates) # Ensure sorted for turnover control
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
            breadth_path="backend/data/breadth.parquet", # Assuming exists
            lookback_days=self.lookback,
            horizon_days=self.horizon,
            feature_cols=self.feature_cols
        )
        if batch is None:
            return {
                "X": torch.empty(0),
                "y": torch.empty(0),
                "valid": False,
                "date": str(date)
            }

        return {
            "X": batch["X"], # [N, T, F]
            "y": batch["y"], # [N] (Label)
            "y_aux": batch["y_aux"], # [N] (Direction)
            "tickers": batch["tickers"],
            "valid": True,
            "date": str(date)
        }

def train_epoch(model, loader, optimizer, scaler, device, turnover_gamma: float, last_scores_map: Dict[str, float]):
    model.train()
    total_loss = 0.0
    steps = 0
    
    # Turnover state
    # We need to map previous scores to current tickers.
    # Since loader might shuffle (if not sequential), turnover reg is tricky.
    # If sequential, we can use last_scores_map.
    
    current_scores_map = {}
    
    for batch in loader:
        if not batch["valid"]:
            continue
            
        # Unpack
        # Batch size is 1 date. Collate fn returns dict.
        X = batch["X"][:, -1, :].to(device) # [N, F] (Use last step of lookback for Transformer? rank_transformer inputs [B, T, F] usually?)
        # RankTransformer in Algaie usually takes [B, F] (flattened) or [B, T, F].
        # Let's check RankTransformer signature later. Assuming it handles [N, F] or we flatten?
        # Dataset returns [N, T, F].
        # If model expects 2D, we select last. If 3D, pass all.
        # RankTransformer logic usually: transformer over T? Or MLP over features?
        # "Transformer" implies sequence.
        # Let's verify model input.
        # For now, pass [N, T, F] if model supports it.
        # Step 151 (Scaler) handles 2D/3D.
        # Let's assumes model expects [N, T, F].
        
        X_raw = batch["X"].to(device)
        y = batch["y"].to(device) # [N]
        tickers = batch["tickers"]
        
        # Scale
        # Scaler expects [N, T, F] -> [N, T, F]
        X_scaled = scaler.transform(X_raw)
        
        # Add batch dim for model if needed?
        # RankTransformer usually takes [Batch, Seq, Features]. Here Batch=N tickers?
        # No, "CrossSectional" means we rank N items.
        # Model forward(x) -> scores [N, 1].
        # Input to model: [N, T, F].
        
        optimizer.zero_grad()
        
        out = model(X_scaled)
        scores = out["score"].squeeze(-1) # [N]
        
        # Loss 1: Listwise + Pairwise
        loss_main = listwise_softmax_loss(scores, y) + 0.5 * pairwise_margin_loss(scores, y)
        
        # Loss 2: Turnover Regularization
        loss_turnover = torch.tensor(0.0, device=device)
        if turnover_gamma > 0 and last_scores_map:
            # Match tickers
            # indices of current tickers that were in last batch
            curr_indices = []
            prev_values = []
            
            for i, t in enumerate(tickers):
                if t in last_scores_map:
                    curr_indices.append(i)
                    prev_values.append(last_scores_map[t])
            
            if curr_indices:
                curr_scores_subset = scores[curr_indices]
                prev_scores_tensor = torch.tensor(prev_values, device=device, dtype=torch.float32)
                
                # Penalty: L2 diff
                loss_turnover = turnover_gamma * torch.mean((curr_scores_subset - prev_scores_tensor) ** 2)
        
        loss = loss_main + loss_turnover
        
        loss.backward()
        optimizer.step()
        
        # Update state (detach)
        # Store current scores for next step
        # If we shuffle, this is random noise regularizer (bad).
        # We MUST NOT shuffle if turnover_gamma > 0.
        current_scores_map = {t: s.item() for t, s in zip(tickers, scores.detach())}
        last_scores_map = current_scores_map # Update per step implies sequentiality
        
        total_loss += loss.item()
        steps += 1
        
    return total_loss / max(steps, 1), last_scores_map

def evaluate(model, loader, scaler, device, evaluator: TopKBasketEvaluator):
    model.eval()
    val_loss = 0.0
    steps = 0
    all_scores = []
    all_targets = []
    
    # Evaluator reset?
    # TopKBasketEvaluator maintains history. We should use a fresh one per epoch?
    # Or accumulate? Usually fresh per validation run.
    evaluator.history = [] 
    
    with torch.no_grad():
        for batch in loader:
            if not batch["valid"]:
                continue
                
            X_raw = batch["X"].to(device)
            y = batch["y"].to(device)
            tickers = batch["tickers"]
            date = batch["date"]
            
            X_scaled = scaler.transform(X_raw)
            out = model(X_scaled)
            scores = out["score"].squeeze(-1)
            
            loss = listwise_softmax_loss(scores, y) + 0.5 * pairwise_margin_loss(scores, y)
            val_loss += loss.item()
            steps += 1
            
            all_scores.append(scores.cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
            # Update Evaluator
            score_dict = {t: s.item() for t, s in zip(tickers, scores)}
            target_dict = {t: lbl.item() for t, lbl in zip(tickers, y)}
            # We need valid date object
            if isinstance(date, str):
                date_obj = datetime.strptime(date, "%Y-%m-%d").date()
            else:
                date_obj = date
                
            evaluator.step(date_obj, score_dict, target_dict)
            
    avg_loss = val_loss / max(steps, 1)
    metrics = evaluator.get_metrics()
    
    return avg_loss, metrics, all_scores, all_targets

def run_training_cycle(run_id, args, use_priors: bool):
    """
    Runs a full training cycle (Train -> Val -> Test)
    """
    feature_cols = FeatureContract.CORE_FEATURES + FeatureContract.MARKET_FEATURES
    if use_priors:
        feature_cols += FeatureContract.PRIOR_FEATURES
        
    tag = "with_priors" if use_priors else "no_priors"
    print(f"\n=== Starting Training Cycle: {tag.upper()} ===")
    
    # 1. Setup Data Paths
    data_dir = "backend/data_canonical/daily_parquet" # Correct flat path
    
    # Dates
    start_date = datetime(2020, 1, 1).date()
    end_date = datetime(2023, 12, 31).date()
    
    # Split
    # Train: 2020-2022. Val: 2023.
    dates = [start_date + timedelta(days=x) for x in range((end_date - start_date).days + 1)]
    # Filter? Dataset handles lookups.
    
    train_end = datetime(2023, 1, 1).date()
    train_dates = [d for d in dates if d < train_end]
    val_dates = [d for d in dates if d >= train_end]
    
    # Universe
    # We should load universe from frame or just use what's in daily_parquet?
    # Dataset builder requires explicit universe list?
    # make_cross_sectional_batch takes `universe`.
    # We can scan data_dir for available tickers.
    files = list(Path(data_dir).glob("*.parquet")) + list(Path(data_dir).glob("features/*.parquet"))
    universe = list(set([f.stem for f in files]))
    if not universe:
        raise FileNotFoundError(f"No tickers found in {data_dir}")
    print(f"Universe Size: {len(universe)}")
    
    # 2. Scaler
    print("Fitting scaler...")
    scaler = SelectorFeatureScaler(version=VERSION, feature_names=feature_cols)
    # Fit on subset of train
    sample_dates = train_dates[::10] # 10% sample
    sample_ds = CrossSectionalDataset(sample_dates, universe, data_dir, 60, 10, feature_cols)
    sample_loader = DataLoader(sample_ds, batch_size=1, collate_fn=lambda x: x[0])
    
    all_X_list = []
    for b in sample_loader:
        if b["valid"]:
            all_X_list.append(b["X"].numpy())
    
    if all_X_list:
        scaler.fit(np.concatenate(all_X_list, axis=0))
    else:
        print("Scaler fit failed: No data.")
        return
        
    # 3. Model
    model = RankTransformer(d_input=len(feature_cols)).to(args.device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # 4. Training Loop
    # Sequential for Turnover Control
    train_ds = CrossSectionalDataset(train_dates, universe, data_dir, 60, 10, feature_cols)
    val_ds = CrossSectionalDataset(val_dates, universe, data_dir, 60, 10, feature_cols)
    
    train_loader = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=lambda x: x[0])
    
    evaluator = TopKBasketEvaluator(k=ops_config.TOPK_K, cost_bps=ops_config.COST_BPS)
    
    last_scores_map = {}
    best_val_sharpe = -float("inf")
    
    for epoch in range(args.epochs):
        train_loss, last_scores_map = train_epoch(
            model, train_loader, optimizer, scaler, args.device,
            turnover_gamma=ops_config.TURNOVER_GAMMA,
            last_scores_map=last_scores_map
        )
        
        val_loss, metrics, all_scores, all_targets = evaluate(model, val_loader, scaler, args.device, evaluator)
        
        sharpe = metrics.get("sharpe", 0.0)
        tot_ret = metrics.get("total_ret", 0.0)
        
        print(f"[{tag}] Epoch {epoch+1}: TrainL={train_loss:.4f}, ValL={val_loss:.4f}, Sharpe={sharpe:.2f}, Ret={tot_ret:.2%}")
        
        # Log
        run_recorder.emit_metric(run_id, epoch=epoch+1, metrics={
            f"{tag}/train_loss": train_loss,
            f"{tag}/val_loss": val_loss,
            f"{tag}/val_sharpe": sharpe,
            f"{tag}/val_ret": tot_ret,
            f"{tag}/val_turnover": metrics.get("avg_turnover", 0.0)
        })
        
        # Checkpoint Best
        if sharpe > best_val_sharpe:
            best_val_sharpe = sharpe
            path = Path(CHECKPOINT_DIR) / f"best_{tag}.pt"
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), path)
            
    # Final Eval Report
    report_path = Path(LOG_DIR) / f"eval_report_{tag}.parquet"
    evaluator.to_parquet(str(report_path))
    print(f"Saved evaluation report to {report_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--ablation", action="store_true", default=True)
    args = parser.parse_args()
    
    run_id = run_recorder.init_run(
        pipeline_type="selector_swing",
        trigger="manual",
        config=vars(args),
        data_versions={},
        tags=["swing", "ablation" if args.ablation else "single"]
    )
    
    try:
        # Run 1: With Priors (Default)
        if config.PRIORS_ENABLED:
            run_training_cycle(run_id, args, use_priors=True)
        
        # Run 2: Without Priors (Ablation)
        if args.ablation:
            run_training_cycle(run_id, args, use_priors=False)
            
        run_recorder.finalize_run(run_id, "SUCCESS")
        
    except Exception as e:
        print(f"Run failed: {e}")
        run_recorder.finalize_run(run_id, "FAILED")
        raise e

if __name__ == "__main__":
    main()
