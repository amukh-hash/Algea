import os
import sys
import polars as pl
import argparse
from tqdm import tqdm
import pandas as pd

# Ensure backend in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from backend.app.engine.equity_pod import EquityPod
from backend.app.risk.types import ActionType

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--data_dir", default="backend/data/marketframe")
    parser.add_argument("--model_path", default="backend/models/student/student_v1.pt")
    parser.add_argument("--preproc_path", default="backend/models/preproc/preproc_v1.json")
    args = parser.parse_args()
    
    # 1. Load Data (Simulate Feed)
    mf_path = os.path.join(args.data_dir, f"marketframe_{args.ticker}_1m.parquet")
    if not os.path.exists(mf_path):
        print(f"Data not found: {mf_path}")
        return
        
    df = pl.read_parquet(mf_path)
    # Take last 1000 rows
    df = df.tail(1000)
    
    print(f"Starting Paper Trading Simulation for {args.ticker} on {df.height} bars...")
    
    # 2. Init Pod
    try:
        pod = EquityPod(args.ticker, args.model_path, args.preproc_path)
    except Exception as e:
        print(f"Failed to init pod: {e}")
        return

    # 3. Loop
    # We iterate rows.
    # We need to construct tick + breadth dict.
    
    # Pre-fetch breadth columns to avoid slow access?
    # breadth cols are in the DF.
    
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
        
        # Run Pod
        decision = pod.on_tick(tick, breadth)
        
        if decision:
            if decision.action not in (ActionType.NO_NEW_RISK, ActionType.HOLD):
                print(f"[{row['timestamp']}] DECISION: {decision.action} {decision.quantity} | {decision.reason}")
                pod.execute_decision(decision)
                
    # 4. Final State
    print("Final Portfolio State:")
    print(f"Cash: {pod.portfolio.cash}")
    print(f"Positions: {pod.portfolio.positions}")
    print(f"Total Equity: {pod.portfolio.total_equity}")

if __name__ == "__main__":
    main()
