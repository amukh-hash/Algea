import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))

import pandas as pd
import numpy as np
import os
import argparse
from backend.app.ops import bootstrap, pathmap, run_recorder
# import alpaca_trade_api as tradeapi  <-- Removed to avoid import error

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tickers for testing")
    parser.add_argument("--output", default=None)
    parser.add_argument("--macro_output", default=None)
    args = parser.parse_args()
    
    bootstrap.ensure_dirs()
    run_id = run_recorder.init_run(
        pipeline_type="full_pipeline",
        trigger="manual",
        config={
            "start": args.start,
            "end": args.end,
            "limit": args.limit,
        },
        data_versions={"gold": "unknown", "silver": "unknown", "macro": "unknown", "universe": "unknown"},
        tags=["ingest", "ohlcv_raw"],
    )
    run_dir = run_recorder.run_paths.get_run_dir(run_id)
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else outputs_dir / "raw_ohlcv.parquet"
    macro_output_path = Path(args.macro_output) if args.macro_output else outputs_dir / "macro_raw.parquet"
    try:
        run_recorder.set_status(run_id, "RUNNING", stage="ingest", step="generate_raw")
        run_recorder.emit_event(run_id, "ingest", "start", "INFO", "Mocking fetch_data", {"start": args.start, "end": args.end})
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
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(output_path)
        print(f"Written mock equity data to {output_path}")
        run_recorder.register_artifact(
            run_id,
            name="raw_ohlcv",
            type="parquet",
            path=str(output_path),
            tags=["ohlcv", "raw"],
        )
        
        # Mock Macro
        macro_df = pd.DataFrame({'VIXCLS': 20.0 + np.random.randn(len(dates)), 'DGS10': 4.0}, index=dates)
        os.makedirs(os.path.dirname(macro_output_path), exist_ok=True)
        macro_df.to_parquet(macro_output_path)
        print(f"Written mock macro data to {macro_output_path}")
        run_recorder.register_artifact(
            run_id,
            name="macro_raw",
            type="parquet",
            path=str(macro_output_path),
            tags=["macro", "raw"],
        )
        run_recorder.finalize_run(run_id, "PASSED")
    except Exception as exc:
        run_recorder.set_status(
            run_id,
            "FAILED",
            stage="ingest",
            step="error",
            error={"type": type(exc).__name__, "message": str(exc), "traceback": ""},
        )
        run_recorder.finalize_run(run_id, "FAILED")
        raise

if __name__ == "__main__":
    main()
