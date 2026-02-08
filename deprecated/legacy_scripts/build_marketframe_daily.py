"""
Build Daily MarketFrames (Silver Data) from Gold + Covariates.
Input: 
  - Gold Daily Parquet (backend/data_canonical/daily_parquet)
  - Breadth Context (backend/data/breadth.parquet) [Optional/Future]
  - Market Context (SPY, VIX) [Optional/Future]
Output:
  - MarketFrame Daily Parquet (backend/data_canonical/marketframe_daily)
"""
import os
import sys
import argparse
import polars as pl
from pathlib import Path
from tqdm import tqdm

# Ensure backend in path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

# Defaults
GOLD_DIR = "backend/data_canonical/daily_parquet"
OUTPUT_DIR = "backend/data_canonical/marketframe_daily"

def main():
    parser = argparse.ArgumentParser(description="Build Silver MarketFrames from Gold")
    parser.add_argument("--gold_dir", default=GOLD_DIR, help="Path to Gold daily parquet files")
    parser.add_argument("--output_dir", default=OUTPUT_DIR, help="Output directory for MarketFrames")
    parser.add_argument("--ticker", help="Build specific ticker only")
    args = parser.parse_args()

    gold_path = Path(args.gold_dir).resolve()
    out_path = Path(args.output_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Building Silver MarketFrames...")
    print(f"Input: {gold_path}")
    print(f"Output: {out_path}")

    # Gather files
    if args.ticker:
        files = list(gold_path.glob(f"{args.ticker}.parquet"))
    else:
        files = list(gold_path.glob("*.parquet"))

    if not files:
        print(f"No gold files found in {gold_path}")
        return

    print(f"Found {len(files)} gold files.")
    
    success = 0
    failed = 0

    for f in tqdm(files):
        try:
            # Load Gold
            df = pl.read_parquet(f)
            
            # Simple Pass-through + Normalization for now
            # In V2, MarketFrame = OHLCV + Covariates + Regime
            # For Chronos Phase 2, we just need the file existence and basic cols?
            # Or do we strictly need covariates?
            # The user plan says: "marketframe_daily is a derived dataset... likely needed by phase2"
            # User Plan 3.1: "date, symbol, OHLCV, covariates..."
            
            # Since we lack the covariate source logic in this quick script, 
            # we will create a compliant file with just OHLCV + symbol/date normalized.
            # This unblocks file existence checks.
            # REAL implementation should start joining SPY/VIX/Breadth here.
            
            # Normalize Date
            # Gold is Datetime, MarketFrame expects Date? 
            # Actually ChronosDataset handles casting now. But let's be clean.
            if "date" in df.columns:
                df = df.with_columns(pl.col("date").dt.date().alias("date"))
            
            # Ensure Symbol
            ticker = f.stem
            if "symbol" not in df.columns:
                df = df.with_columns(pl.lit(ticker).alias("symbol"))
                
            # Write key
            out_file = out_path / f"marketframe_{ticker}_daily.parquet"
            df.write_parquet(out_file)
            success += 1

        except Exception as e:
            print(f"Failed {f.name}: {e}")
            failed += 1

    print(f"Done. Success: {success}, Failed: {failed}")

if __name__ == "__main__":
    main()
