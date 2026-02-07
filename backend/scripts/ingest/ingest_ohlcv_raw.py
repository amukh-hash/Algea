import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import pandas as pd
import numpy as np
import os
import argparse
from backend.app.ops import bootstrap, pathmap
# import alpaca_trade_api as tradeapi  <-- Removed to avoid import error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tickers for testing")
    parser.add_argument("--output", default="backend/data/artifacts/universe/raw_ohlcv.parquet")
    parser.add_argument("--macro_output", default="backend/data/artifacts/features/macro_raw.parquet")
    args = parser.parse_args()
    
    bootstrap.ensure_dirs()
    
    print(f"Mocking fetch_data for {args.start} to {args.end}...")
    
    # Create Dummy Raw Data
    dates = pd.date_range(start=args.start, end=args.end, freq='B')
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'SPY']
    
    data = []
    for s in symbols:
        for d in dates:
            data.append({
                'date': d,
                'ticker': s, # Legacy format used 'ticker'?
                'open': 100.0 + np.random.randn(),
                'high': 105.0 + np.random.randn(),
                'low': 95.0 + np.random.randn(),
                'close': 102.0 + np.random.randn(),
                'volume': 1000000, # int
                'vwap': 101.0,
                'trade_count': 5000
            })
            
    df = pd.DataFrame(data)
    
    # Ensure output dir
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_parquet(args.output)
    print(f"Written mock equity data to {args.output}")
    
    # Mock Macro
    macro_df = pd.DataFrame({'VIXCLS': 20.0 + np.random.randn(len(dates)), 'DGS10': 4.0}, index=dates)
    os.makedirs(os.path.dirname(args.macro_output), exist_ok=True)
    macro_df.to_parquet(args.macro_output)
    print(f"Written mock macro data to {args.macro_output}")

if __name__ == "__main__":
    main()
