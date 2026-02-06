
import logging
import argparse
import pandas as pd
import os
from tqdm import tqdm
from backend.app.ops import bootstrap, pathmap, config
from backend.app.data.ingest import ohlcv_daily
from backend.app.data.adjustments import adjust_daily_bars

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2014-01-01")
    parser.add_argument("--end", default="2026-02-03")
    parser.add_argument("--source", default="legacy") # 'legacy' reads from raw_ohlcv.parquet
    args = parser.parse_args()
    
    bootstrap.ensure_dirs()
    
    # Source: Read legacy raw_ohlcv if exists, to migrate
    paths = pathmap.get_paths()
    legacy_path = os.path.join(paths.legacy_artifacts_root, "universe/raw_ohlcv.parquet")
    
    if not os.path.exists(legacy_path):
        logger.error(f"Legacy Raw OHLCV not found at {legacy_path}. Cannot backfill.")
        return
        
    logger.info("Reading Legacy Data...")
    df = pd.read_parquet(legacy_path)
    
    # Group by Ticker
    tickers = df['ticker'].unique()
    logger.info(f"Backfilling {len(tickers)} tickers...")
    
    for ticker in tqdm(tickers):
        sub = df[df['ticker'] == ticker].copy()
        
        # Adjust
        # Map legacy cols -> Schema B2
        # Legacy: date, open, high, low, close, volume, ticker
        sub.rename(columns={
            "open": "open_adj", 
            "high": "high_adj", 
            "low": "low_adj", 
            "close": "close_adj"
        }, inplace=True)
        
        # Apply Adjustments logic (computes dollar_vol etc)
        adj_df = adjust_daily_bars(sub)
        
        # Write Partition
        # Filter date range
        mask = (adj_df['date'] >= pd.Timestamp(args.start)) & (adj_df['date'] <= pd.Timestamp(args.end))
        final_df = adj_df[mask].copy()
        
        if not final_df.empty:
            ohlcv_daily.write_ohlcv_partition(ticker, final_df)
            
    logger.info("Backfill Complete.")

if __name__ == "__main__":
    main()
