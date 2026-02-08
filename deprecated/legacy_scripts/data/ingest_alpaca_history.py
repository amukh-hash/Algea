import os
import sys
import time
import logging
import argparse
import requests
import pandas as pd
import polars as pl
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_URL = "https://data.alpaca.markets/v2/stocks"
MAX_WORKERS = 8  # Conservative to avoid rate limits (200/min)
RATE_LIMIT_DELAY = 0.5 # Seconds between requests per worker

def load_env_vars():
    """Load environment variables from .env file if present."""
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path, "r") as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    key, val = line.strip().split("=", 1)
                    os.environ[key] = val.strip('"').strip("'")

def get_headers():
    api_key = os.environ.get("APCA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY_ID") or os.environ.get("ALPACA_API_KEY")
    secret_key = os.environ.get("APCA_API_SECRET_KEY") or os.environ.get("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        # Fallback: check if we just loaded them but os.environ didn't pick up? 
        # load_env_vars() should have set them.
        pass

    if not api_key or not secret_key:
        raise ValueError(f"Alpaca API keys not found. Checked APCA_API_KEY_ID, ALPACA_API_KEY_ID, ALPACA_API_KEY.")
        
    return {
        "APCA-API-KEY-ID": api_key,
        "APCA-API-SECRET-KEY": secret_key,
        "accept": "application/json"
    }

def fetch_ticker_history(symbol, start_date, end_date, headers, feed="sip"):
    """
    Fetch daily bars for a single ticker.
    Returns a list of dicts or None if error/empty.
    """
    all_bars = []
    page_token = None

    params = {
        "start": start_date,
        "end": end_date,
        "timeframe": "1Day",
        "adjustment": "all",
        "limit": 10000,
        "feed": feed,
    }
    
    url = f"{BASE_URL}/{symbol}/bars"
    
    while True:
        if page_token:
            params["page_token"] = page_token
            
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=10)
            
            if resp.status_code == 429:
                logger.warning(f"Rate limited on {symbol}. Sleeping 2s...")
                time.sleep(2)
                continue
            
            if resp.status_code != 200:
                logger.error(f"Error fetching {symbol}: {resp.status_code} - {resp.text}")
                return None
                
            resp.raise_for_status()
            data = resp.json()
            
            bars = data.get("bars", [])
            if not bars:
                break
                
            all_bars.extend(bars)
            
            page_token = data.get("next_page_token")
            if not page_token:
                break
                
            # Nice-to-have delay to be kind to API
            # time.sleep(0.1) 
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None

    return all_bars

def process_ticker(symbol, start_date, end_date, output_dir, headers, feed="sip"):
    """
    Worker function to process a single ticker.
    """
    try:
        # Skip if already downloaded
        out_path = output_dir / f"{symbol}.parquet"
        if out_path.exists():
            return True

        bars = fetch_ticker_history(symbol, start_date, end_date, headers, feed=feed)
        
        if not bars:
            logger.warning(f"No data found for {symbol}")
            return False

        # Convert to DataFrame
        # Alpaca bar format: {'t': '2022-01-03T05:00:00Z', 'o': 177.83, 'h': 182.88, 'l': 177.71, 'c': 182.01, 'v': 104487930, 'n': 763953, 'vw': 180.93}
        df = pd.DataFrame(bars)
        df['symbol'] = symbol # Use 'symbol' to match existing daily_parquet schema
        df = df.rename(columns={
            't': 'date',
            'o': 'open',
            'h': 'high',
            'l': 'low',
            'c': 'close',
            'v': 'volume',
            'n': 'trade_count',
            'vw': 'vwap'
        })
        
        # Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Save to parquet
        out_path = output_dir / f"{symbol}.parquet"
        df.to_parquet(out_path, index=False)
        return True
        
    except Exception as e:
        logger.error(f"Failed to process {symbol}: {e}")
        return False

def merge_to_raw_ohlcv(daily_dir, output_path):
    """
    Merge all daily parquet files into a single raw_ohlcv.parquet file using Pandas (fallback).
    """
    logger.info(f"Merging files from {daily_dir} to {output_path} (Pandas mode)...")
    try:
        daily_files = list(daily_dir.glob("*.parquet"))
        if not daily_files:
            logger.warning("No files to merge.")
            return

        dfs = []
        for f in daily_files:
            try:
                df = pd.read_parquet(f)
                
                # Renaissance
                if "symbol" in df.columns:
                    df = df.rename(columns={"symbol": "ticker"})
                
                if "ticker" not in df.columns:
                    # Try to infer from filename if really needed, but skip for now
                    continue
                
                if "date" not in df.columns:
                    continue
                
                # Standardize columns
                cols_map = {
                    "volume": "Int64", 
                    "trade_count": "Int64",
                    "open": "float64", 
                    "high": "float64", 
                    "low": "float64", 
                    "close": "float64", 
                    "vwap": "float64"
                }
                
                for c, dtype in cols_map.items():
                    if c not in df.columns:
                        df[c] = pd.NA
                    # Pandas casting
                    df[c] = df[c].astype(dtype)
                
                # Keep only relevant columns to save memory
                keep_cols = ["date", "ticker"] + list(cols_map.keys())
                df = df[keep_cols]
                
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Failed to read {f}: {e}")
            
        if not dfs:
            return

        # Concat
        full_df = pd.concat(dfs, ignore_index=True)
        full_df = full_df.sort_values(["ticker", "date"])
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_df.to_parquet(output_path, index=False)
        logger.info("Merge complete.")
    except Exception as e:
        logger.error(f"Merge failed: {e}")

def main():
    parser = argparse.ArgumentParser(description="Ingest Alpaca History")
    parser.add_argument("--start", default="2006-01-01")
    parser.add_argument("--end", default="2025-12-31")
    parser.add_argument("--metadata", default="backend/data_canonical/security_master.parquet")
    parser.add_argument("--out_dir", default="backend/data_canonical/daily_parquet")
    parser.add_argument("--merge_out", default="backend/data/artifacts/universe/raw_ohlcv.parquet")
    parser.add_argument("--feed", default="sip", choices=["sip", "iex"], help="Alpaca data feed (default: sip)")
    parser.add_argument("--test_mode", action="store_true", help="Run on first 10 tickers only")
    args = parser.parse_args()

    load_env_vars()
    headers = get_headers()
    
    # Init output dir
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load Universe
    if not os.path.exists(args.metadata):
        logger.error(f"Metadata file not found: {args.metadata}")
        sys.exit(1)
        
    univ_df = pd.read_parquet(args.metadata)
    # Check for 'symbol' or 'ticker'
    col = 'symbol' if 'symbol' in univ_df.columns else 'ticker'
    tickers = univ_df[col].unique().tolist()
    
    if args.test_mode:
        tickers = tickers[:10]
        logger.info("TEST MODE: Processing only 10 tickers.")
        
    logger.info(f"Starting ingestion for {len(tickers)} tickers from {args.start} to {args.end}")
    
    # Process in parallel
    count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_ticker, t, args.start, args.end, output_dir, headers, feed=args.feed): t
            for t in tickers
        }
        
        for future in as_completed(futures):
            t = futures[future]
            try:
                success = future.result()
                if success:
                    count += 1
                if count % 100 == 0:
                    logger.info(f"Processed {count} tickers...")
            except Exception as exc:
                logger.error(f"{t} generated an exception: {exc}")

    logger.info(f"Ingestion complete. Downloaded {count}/{len(tickers)} tickers.")
    
    # Merge step
    merge_to_raw_ohlcv(output_dir, Path(args.merge_out))

if __name__ == "__main__":
    main()
