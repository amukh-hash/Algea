import os
import sys
import argparse
import json
import polars as pl
from datetime import datetime, timedelta

# Ensure backend in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../.."))

from backend.app.options.data.providers.mock import MockIVProvider
from backend.app.core.config import OPTIONS_DATA_VERSION

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start_date", default="2023-01-01")
    parser.add_argument("--end_date", default="2023-12-31")
    parser.add_argument("--dte", type=int, default=30)
    parser.add_argument("--output_dir", default="backend/data/options/iv")
    args = parser.parse_args()
    
    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")
    
    print(f"Ingesting IV for {args.ticker} from {start_dt} to {end_dt} (DTE: {args.dte})...")
    
    provider = MockIVProvider() # Could add provider selection logic
    
    snapshots = provider.get_iv_history(args.ticker, start_dt, end_dt, args.dte)
    
    if not snapshots:
        print("No data found.")
        return
        
    # Convert to Polars DataFrame
    rows = []
    for s in snapshots:
        rows.append({
            "timestamp": s.timestamp,
            "ticker": s.ticker,
            "dte": s.dte,
            "atm_iv": s.atm_iv,
            "iv_rank": s.iv_rank,
            "iv_percentile": s.iv_percentile
        })
        
    df = pl.DataFrame(rows)
    
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{args.ticker}.parquet")
    
    # If exists, append? For now overwrite or concat.
    if os.path.exists(out_path):
        existing_df = pl.read_parquet(out_path)
        # Filter out overlapping dates? Or just append and unique.
        df = pl.concat([existing_df, df]).unique(subset=["timestamp", "dte"]).sort("timestamp")
        
    df.write_parquet(out_path)
    print(f"Wrote {df.height} rows to {out_path}")
    
    # Write metadata
    meta_path = os.path.join(args.output_dir, f"{args.ticker}_metadata.json")
    metadata = {
        "ticker": args.ticker,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "options_data_version": OPTIONS_DATA_VERSION,
        "options_schema_version": 1,
        "generated_at": datetime.now().isoformat()
    }
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Wrote metadata to {meta_path}")

if __name__ == "__main__":
    main()
