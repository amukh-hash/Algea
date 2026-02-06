import os
import sys
from pathlib import Path

# Set Config for Hybrid Monitor Mode
os.environ["ENABLE_OPTIONS"] = "true"
os.environ["OPTIONS_MODE"] = "monitor" # No orders
os.environ["OPTIONS_SEED"] = "42"

import polars as pl
import argparse
from tqdm import tqdm
from datetime import datetime

# Ensure backend in path
repo_root = next(p.parent for p in Path(__file__).resolve().parents if p.name == "backend")
sys.path.append(str(repo_root))

from backend.app.engine.equity_pod import EquityPod
from backend.app.engine.options_pod import OptionsPod
from backend.app.risk.types import ActionType
from backend.app.options.gate.context import OptionsContext
from backend.app.core.config import EQUITIES_MODE

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--data_dir", default="backend/data/marketframe")
    parser.add_argument("--model_path", default="backend/models/student/student_v1.pt")
    parser.add_argument("--preproc_path", default="backend/models/preproc/preproc_v1.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    
    use_mock = (EQUITIES_MODE == "mock") or args.dry_run

    # 1. Load Data (Simulate Feed)
    mf_path = os.path.join(args.data_dir, f"marketframe_{args.ticker}_1m.parquet")
    
    # Mock data creation if not exists (for smoke test)
    if args.dry_run and not os.path.exists(mf_path):
        print("Dry run: Generating dummy data...")
        # Create dummy MF
        dates = [datetime(2023, 1, 1) for _ in range(200)]
        df = pl.DataFrame({
            "timestamp": dates,
            "open": [150.0]*200,
            "high": [151.0]*200,
            "low": [149.0]*200,
            "close": [150.0]*200,
            "volume": [1000.0]*200,
            "ad_line": [1000.0]*200,
            "bpi": [60.0]*200
        })
    elif not os.path.exists(mf_path):
        print(f"Data not found: {mf_path}")
        return
    else:
        df = pl.read_parquet(mf_path)
        df = df.tail(1000)

    print(f"Starting Hybrid Monitor Simulation for {args.ticker}...")

    # 2. Init Pods
    try:
        equity_pod = EquityPod(
            args.ticker, 
            args.model_path, 
            args.preproc_path, 
            use_mock_runner=use_mock
        )
        options_pod = OptionsPod()
    except Exception as e:
        print(f"Failed to init pods: {e}")
        # If dry run, maybe models are missing.
        if args.dry_run:
             print("Dry run: Mocking pods failure ignored.")
             return
        return

    # 3. Loop
    rows = df.iter_rows(named=True)

    for row in tqdm(rows, total=df.height):
        # Extract tick
        tick = {
            "timestamp": row["timestamp"],
            "open": row["open"],
            "high": row["high"],
            "low": row["low"],
            "close": row["close"],
            "volume": row["volume"]
        }

        breadth = {
            "ad_line": row["ad_line"],
            "bpi": row["bpi"]
        }

        # Run Equity Pod (computes signal)
        eq_decision = equity_pod.on_tick(tick, breadth)

        if eq_decision and eq_decision.action not in (ActionType.NO_NEW_RISK, ActionType.HOLD):
            print(f"[EQUITY] {row['timestamp']} DECISION: {eq_decision.action} {eq_decision.quantity} | {eq_decision.reason}")
            equity_pod.execute_decision(eq_decision)
            
        # Run Options Pod (using shared signal)
        signal = equity_pod.last_signal
        if signal:
            ctx = OptionsContext(
                ticker=args.ticker,
                timestamp=row["timestamp"],
                underlying_price=row["close"],
                student_signal=signal,
                breadth=breadth,
                posture="NORMAL" # Could derive from RiskManager
            )
            
            opt_decision = options_pod.on_signal(ctx)
            if opt_decision:
                print(f"[OPTIONS] {row['timestamp']} DECISION: {opt_decision.action} {opt_decision.candidate.short_strike}/{opt_decision.candidate.long_strike} | {opt_decision.reason}")
                options_pod.execute(opt_decision)

    print("Simulation Complete.")

if __name__ == "__main__":
    main()
