import os
import sys
from pathlib import Path
import argparse
import polars as pl
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure backend in path
repo_root = next(p.parent for p in Path(__file__).resolve().parents if p.name == "backend")
sys.path.append(str(repo_root))

from backend.app.models.student_inference import StudentRunner
from backend.app.models.baseline import BaselineMLP # or dynamic import
from backend.app.data import windows, splits
from backend.app.eval import metrics, promotion_gate, reporting

def evaluate(args):
    # 1. Load Data
    # For evaluation, we ideally use a holdout set.
    # We load ONE ticker for now (AAPL) or loop?
    # Let's do AAPL.
    ticker = "AAPL"
    mf_path = os.path.join(args.data_dir, f"marketframe_{ticker}_1m.parquet")
    if not os.path.exists(mf_path):
        print(f"Data not found: {mf_path}")
        return

    df = pl.read_parquet(mf_path)
    
    # Split? Let's take last 20% for eval
    # Or just use the whole file if it's designated test file.
    # We'll use splits.
    train, val, test = splits.get_time_splits(df)
    eval_df = test if test.height > 0 else val
    
    print(f"Evaluating on {eval_df.height} rows...")
    
    # 2. Init Runner
    runner = StudentRunner(args.model_path, args.preproc_path, device=args.device)
    
    # 3. Create Dataset
    # We need inputs for model.
    # Runner does transform internally for single inference, but for bulk eval?
    # Runner.infer takes a DataFrame window.
    # Doing that loop is slow in python.
    # Better: Transform whole DF, create tensor dataset, batch infer.
    # But Runner wraps the model.
    # We can use runner.model directly if we are careful, or add a method `infer_batch`.
    # Let's use runner.model directly for efficiency here, duplicating some logic from runner.
    # Ideally Runner has `predict_batch`.
    
    preproc = runner.preproc
    df_trans = preproc.transform(eval_df)
    if "close" not in df_trans.columns:
        df_trans = df_trans.with_columns(eval_df["close"])
        
    cols = ["log_ret", "volume_norm", "ad_line_norm", "bpi_norm"]
    ds = windows.SwingWindowDataset(df_trans, cols, lookback=128, stride=15) # Stride 15m
    
    if len(ds) == 0:
        print("Dataset empty.")
        return
        
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    
    # 4. Inference Loop
    all_preds_1d = {"0.05": [], "0.50": [], "0.95": []}
    all_targets_1d = []
    
    runner.model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(args.device)
            # y["1D"] is scalar target
            all_targets_1d.extend(y["1D"].numpy())
            
            out = runner.model(x) # (B, H, Q)
            # 1D is index 0
            # Quantiles: 0=0.05, 1=0.50, 2=0.95
            
            preds = out[:, 0, :].cpu().numpy()
            all_preds_1d["0.05"].extend(preds[:, 0])
            all_preds_1d["0.50"].extend(preds[:, 1])
            all_preds_1d["0.95"].extend(preds[:, 2])

    # 5. Compute Metrics
    y_true = np.array(all_targets_1d)
    q_dict = {k: np.array(v) for k, v in all_preds_1d.items()}
    
    results = metrics.compute_metrics(y_true, q_dict)
    print("Metrics:", results)
    
    # 6. Promotion Check
    gate = promotion_gate.PromotionGate()
    # Baseline? We assume some baseline metrics or load from prev report.
    # For Phase 1, use hardcoded baseline
    baseline = {"accuracy": 0.50, "width_90": 0.10} # Dummy
    
    passed, reasons = gate.check(results, baseline)
    
    # 7. Report
    reporting.save_report(results, passed, reasons, args.report_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="backend/models/student/student_v1.pt")
    parser.add_argument("--preproc_path", default="backend/models/preproc/preproc_v1.json")
    parser.add_argument("--data_dir", default="backend/data/marketframe")
    parser.add_argument("--report_path", default="backend/models/student/eval_report.json")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    
    evaluate(args)
