import os
import logging
import argparse
import time
from datetime import datetime, timedelta
import pandas as pd
import alpaca_trade_api as tradeapi
import requests
from backend.app.core import config
from backend.app.ops import bootstrap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
FRED_API_KEY = os.getenv("FRED_API_KEY")

class AlpacaIngestor:
    def __init__(self):
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise ValueError("ALPACA_API_KEY/SECRET_KEY missing")
        self.api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, api_version='v2')

    def get_active_assets(self) -> list:
        logger.info("Fetching active assets from Alpaca...")
        assets = self.api.list_assets(status='active', asset_class='us_equity')
        # Filter for tradable and marginable?
        # Ideally filter by exchange (NYSE, NASDAQ, AMEX, ARCA)
        allowed_exchanges = ['NYSE', 'NASDAQ', 'AMEX', 'ARCA']
        us_equities = [a for a in assets if a.exchange in allowed_exchanges and a.tradable]
        logger.info(f"Found {len(us_equities)} active US equities.")
        return [a.symbol for a in us_equities]

    def fetch_bars(self, symbols: list, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetches daily bars for a list of symbols.
        Batches requests to avoid URL length limits/timeouts.
        """
        batch_size = 200 # Alpaca limit per request
        all_bars = []
        
        # Parse dates
        start = pd.Timestamp(start_date).isoformat()
        end = pd.Timestamp(end_date).isoformat()
        
        total = len(symbols)
        for i in range(0, total, batch_size):
            batch = symbols[i : i + batch_size]
            logger.info(f"Fetching batch {i//batch_size + 1}/{(total//batch_size)+1} ({len(batch)} symbols)...")
            
            try:
                # get_bars return is a BarSet, need to iterate
                # Using RFC-3339 format
                bars = self.api.get_bars(batch, tradeapi.TimeFrame.Day, start=start, end=end, adjustment='raw').df
                if not bars.empty:
                    # Reset index to have 'timestamp' as column, or keep date index
                    # Alpaca 'df' property usually has MultiIndex (symbol, timestamp) or just timestamp with symbol column?
                    # get_bars().df usually returns timestamp index and 'symbol' column
                    all_bars.append(bars)
                time.sleep(0.5) # Rate limit politeness
            except Exception as e:
                logger.error(f"Error fetching batch: {e}")
                
        if not all_bars:
            return pd.DataFrame()
            
        final_df = pd.concat(all_bars)
        return final_df

class FredIngestor:
    def __init__(self):
        if not FRED_API_KEY:
            logger.warning("FRED_API_KEY missing. Macro data fetch will fail.")
    
    def fetch_series(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        if not FRED_API_KEY:
            return pd.DataFrame()
            
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": FRED_API_KEY,
            "file_type": "json",
            "observation_start": start_date,
            "observation_end": end_date
        }
        
        try:
            r = requests.get(url, params=params)
            r.raise_for_status()
            data = r.json()
            obs = data.get("observations", [])
            df = pd.DataFrame(obs)
            if df.empty:
                return df
                
            df['date'] = pd.to_datetime(df['date'])
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            df.set_index('date', inplace=True)
            df.rename(columns={'value': series_id}, inplace=True)
            return df[[series_id]]
            
        except Exception as e:
            logger.error(f"FRED fetch failed for {series_id}: {e}")
            return pd.DataFrame()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default=config.TRAIN_START_DATE)
    parser.add_argument("--end", default=config.TRAIN_END_DATE)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of tickers for testing")
    parser.add_argument("--output", default="backend/data/artifacts/universe/raw_ohlcv.parquet")
    parser.add_argument("--macro_output", default="backend/data/artifacts/features/macro_raw.parquet")
    args = parser.parse_args()
    
    # Bootstrap Directories
    bootstrap.ensure_dirs()
    
    # 1. Macro Data
    logger.info("Starting Macro Ingestion...")
    fred = FredIngestor()
    vix = fred.fetch_series("VIXCLS", args.start, args.end)
    tnx = fred.fetch_series("DGS10", args.start, args.end)
    
    if not vix.empty and not tnx.empty:
        macro_df = vix.join(tnx, how='outer').sort_index()
        # Save
        os.makedirs(os.path.dirname(args.macro_output), exist_ok=True)
        macro_df.to_parquet(args.macro_output)
        logger.info(f"Saved Macro data to {args.macro_output}")
    else:
        logger.warning("Skipping Macro save due to missing data.")

    # 2. Equity Data
    logger.info("Starting Equity Ingestion...")
    alpaca = AlpacaIngestor()
    symbols = alpaca.get_active_assets()
    
    if args.limit > 0:
        symbols = symbols[:args.limit]
        logger.info(f"Limiting to first {args.limit} symbols.")
        
    df = alpaca.fetch_bars(symbols, args.start, args.end)
    
    if not df.empty:
        # Standardize Columns
        # Expect: date, ticker, open, high, low, close, volume
        # Alpaca df has index=timestamp, column=symbol (if single?) or 'symbol' column
        
        # Typically:
        # Index: timestamp
        # Columns: open, high, low, close, volume, trade_count, vwap, symbol
        
        df.reset_index(inplace=True)
        # Rename 'timestamp' to 'date' if needed
        if 'timestamp' in df.columns:
            df.rename(columns={'timestamp': 'date'}, inplace=True)
            
        # Ensure ticker column exists
        if 'symbol' in df.columns:
            df.rename(columns={'symbol': 'ticker'}, inplace=True)
            
        # Lowercase columns? 
        df.columns = [c.lower() for c in df.columns]
        
        # Save
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_parquet(args.output)
        logger.info(f"Saved Equity data to {args.output} ({len(df)} rows)")
    else:
        logger.warning("No equity data fetched.")

if __name__ == "__main__":
    main()
