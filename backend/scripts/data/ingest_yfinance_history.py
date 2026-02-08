"""
Ingest historical OHLCV data from Yahoo Finance (2006-2018) to backfill the gap.
Uses yfinance library with parallel processing.
"""
import argparse
import logging
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import yfinance as yf
import polars as pl
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def fetch_ticker_data(ticker: str, start_date: str, end_date: str, output_dir: Path) -> tuple[str, bool]:
    """
    Fetch OHLCV data for a single ticker from Yahoo Finance.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Directory to save parquet files
        
    Returns:
        Tuple of (ticker, success_flag)
    """
    output_file = output_dir / f"{ticker}.parquet"
    
    # Skip if already exists
    if output_file.exists():
        logger.info(f"Skipping {ticker} (already exists)")
        return ticker, True
    
    try:
        # Fetch data from Yahoo Finance
        data = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=False,  # Get unadjusted prices
            actions=False  # Don't need dividends/splits
        )
        
        if data.empty:
            logger.warning(f"No data for {ticker}")
            return ticker, False
        
        # Reset index to get date as column
        data = data.reset_index()
        
        # Flatten multi-level columns if present (yfinance returns tuples for single ticker)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]
        
        # Ensure column names are lowercase strings
        data.columns = [str(col).lower() for col in data.columns]
        
        # Yahoo Finance schema: Date, Open, High, Low, Close, Adj Close, Volume
        # Our schema: date, ticker, open, high, low, close, volume, vwap, trade_count
        
        # Add ticker column
        data['ticker'] = ticker
        
        # Calculate VWAP (approximate using OHLC average)
        # Yahoo doesn't provide true VWAP, so we approximate
        data['vwap'] = (data['open'] + data['high'] + data['low'] + data['close']) / 4.0
        
        # Yahoo doesn't provide trade_count, set to None
        data['trade_count'] = None
        
        # Select and reorder columns
        data = data[['date', 'ticker', 'open', 'high', 'low', 'close', 'volume', 'vwap', 'trade_count']]
        
        # Convert to Polars for consistency
        pl_df = pl.from_pandas(data)
        
        # Ensure correct types
        pl_df = pl_df.with_columns([
            pl.col('date').cast(pl.Datetime),
            pl.col('open').cast(pl.Float64),
            pl.col('high').cast(pl.Float64),
            pl.col('low').cast(pl.Float64),
            pl.col('close').cast(pl.Float64),
            pl.col('volume').cast(pl.Int64),
            pl.col('vwap').cast(pl.Float64),
            pl.col('trade_count').cast(pl.Int64)
        ])
        
        # Save to parquet
        pl_df.write_parquet(output_file)
        logger.info(f"✓ {ticker}: {len(pl_df)} rows")
        return ticker, True
        
    except Exception as e:
        logger.error(f"✗ {ticker}: {e}")
        return ticker, False


def main():
    parser = argparse.ArgumentParser(description="Ingest Yahoo Finance History")
    parser.add_argument("--start_date", default="2006-01-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", default="2018-07-16", help="End date (YYYY-MM-DD)")
    parser.add_argument("--max_workers", type=int, default=10, help="Max parallel workers")
    args = parser.parse_args()
    
    # Directories
    data_dir = Path("backend/data_canonical/daily_parquet_yfinance")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Load security master
    security_master_path = Path("backend/data_canonical/security_master.parquet")
    if not security_master_path.exists():
        logger.error(f"Security master not found: {security_master_path}")
        return
    
    security_master = pl.read_parquet(security_master_path)
    tickers = security_master['symbol'].to_list()
    
    logger.info(f"Starting Yahoo Finance ingestion: {args.start_date} to {args.end_date}")
    logger.info(f"Total tickers: {len(tickers)}")
    logger.info(f"Output directory: {data_dir}")
    logger.info(f"Max workers: {args.max_workers}")
    
    # Process tickers in parallel
    succeeded = []
    failed = []
    
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = {
            executor.submit(fetch_ticker_data, ticker, args.start_date, args.end_date, data_dir): ticker
            for ticker in tickers
        }
        
        for i, future in enumerate(as_completed(futures), 1):
            ticker, success = future.result()
            if success:
                succeeded.append(ticker)
            else:
                failed.append(ticker)
            
            if i % 50 == 0:
                logger.info(f"Progress: {i}/{len(tickers)} ({i/len(tickers)*100:.1f}%)")
    
    logger.info(f"Ingestion complete!")
    logger.info(f"  Success: {len(succeeded)}")
    logger.info(f"  Failed: {len(failed)}")
    
    if failed:
        logger.info(f"Failed tickers: {', '.join(failed[:20])}{'...' if len(failed) > 20 else ''}")


if __name__ == "__main__":
    main()
