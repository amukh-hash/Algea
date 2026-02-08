"""
Merge Yahoo Finance (2006-2018) and Alpaca (2018-2025) data into unified raw_ohlcv.parquet
"""
import logging
from pathlib import Path
import polars as pl
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def merge_all_sources(yfinance_dir, alpaca_dir, output_path, gap_dir=None):
    """
    Merge Yahoo Finance, Alpaca gap, and Alpaca data, deduplicating on overlap.

    Strategy:
    - Yahoo Finance: 2006-01-01 to 2018-07-16 (exclusive of end)
    - Gap (SIP): 2018-07-17 to 2020-07-27
    - Alpaca: 2020-07-27 onwards
    """
    logger.info("Starting unified merge...")
    
    # Load YFinance data (2006-2018)
    logger.info(f"Loading Yahoo Finance data from {yfinance_dir}...")
    yfinance_files = list(yfinance_dir.glob("*.parquet"))
    logger.info(f"  Found {len(yfinance_files)} ticker files")
    
    yf_dfs = []
    for f in yfinance_files:
        try:
            df = pl.read_parquet(f)
            # Keep only data before 2018-07-17
            df = df.filter(pl.col("date") < pl.datetime(2018, 7, 17))
            
            # Ensure consistent schema
            df = df.select([
                pl.col("date").cast(pl.Datetime),
                pl.col("ticker"),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Int64),
                pl.col("vwap").cast(pl.Float64),
                pl.col("trade_count").cast(pl.Int64),
            ])
            
            yf_dfs.append(df)
        except Exception as e:
            logger.warning(f"Error reading {f.name}: {e}")
    
    if yf_dfs:
        yf_combined = pl.concat(yf_dfs, how="vertical_relaxed")
        logger.info(f"  Yahoo Finance: {len(yf_combined)} rows, {yf_combined['date'].min()} to {yf_combined['date'].max()}")
    else:
        yf_combined = None
        logger.warning("  No Yahoo Finance data loaded")
    
    # Load Alpaca data (2018-2025)
    logger.info(f"Loading Alpaca data from {alpaca_dir}...")
    alpaca_files = list(alpaca_dir.glob("*.parquet"))
    logger.info(f"  Found {len(alpaca_files)} ticker files")
    
    alpaca_dfs = []
    for f in alpaca_files:
        try:
            df = pl.read_parquet(f)
            
            # Ensure 'ticker' column (Alpaca uses 'symbol')
            if "symbol" in df.columns:
                df = df.rename({"symbol": "ticker"})
            
            # Ensure consistent schema (same order and types as YFinance)
            df = df.select([
                pl.col("date").cast(pl.Datetime),
                pl.col("ticker"),
                pl.col("open").cast(pl.Float64),
                pl.col("high").cast(pl.Float64),
                pl.col("low").cast(pl.Float64),
                pl.col("close").cast(pl.Float64),
                pl.col("volume").cast(pl.Int64),
                pl.col("vwap").cast(pl.Float64),
                pl.col("trade_count").cast(pl.Int64),
            ])
            
            alpaca_dfs.append(df)
        except Exception as e:
            logger.warning(f"Error reading {f.name}: {e}")
    
    if alpaca_dfs:
        alpaca_combined = pl.concat(alpaca_dfs, how="vertical_relaxed")
        logger.info(f"  Alpaca: {len(alpaca_combined)} rows, {alpaca_combined['date'].min()} to {alpaca_combined['date'].max()}")
    else:
        alpaca_combined = None
        logger.warning("  No Alpaca data loaded")
    
    # Load gap data (2018-07 to 2020-07) if provided
    gap_combined = None
    if gap_dir and gap_dir.exists():
        logger.info(f"Loading gap data from {gap_dir}...")
        gap_files = list(gap_dir.glob("*.parquet"))
        logger.info(f"  Found {len(gap_files)} ticker files")

        gap_dfs = []
        for f in gap_files:
            try:
                df = pl.read_parquet(f)
                if "symbol" in df.columns:
                    df = df.rename({"symbol": "ticker"})
                df = df.select([
                    pl.col("date").cast(pl.Datetime),
                    pl.col("ticker"),
                    pl.col("open").cast(pl.Float64),
                    pl.col("high").cast(pl.Float64),
                    pl.col("low").cast(pl.Float64),
                    pl.col("close").cast(pl.Float64),
                    pl.col("volume").cast(pl.Int64),
                    pl.col("vwap").cast(pl.Float64),
                    pl.col("trade_count").cast(pl.Int64),
                ])
                gap_dfs.append(df)
            except Exception as e:
                logger.warning(f"Error reading gap {f.name}: {e}")
        if gap_dfs:
            gap_combined = pl.concat(gap_dfs, how="vertical_relaxed")
            logger.info(f"  Gap: {len(gap_combined)} rows, {gap_combined['date'].min()} to {gap_combined['date'].max()}")

    # Combine all sources
    logger.info("Combining data sources...")
    parts = [p for p in [yf_combined, gap_combined, alpaca_combined] if p is not None]
    if not parts:
        logger.error("No data to merge!")
        return
    combined = pl.concat(parts, how="vertical_relaxed")

    # Deduplicate on (ticker, date) keeping last (Alpaca preferred over YFinance)
    combined = combined.unique(subset=["ticker", "date"], keep="last")
    
    # Sort by ticker, date
    logger.info("Sorting and saving...")
    combined = combined.sort(["ticker", "date"])
    
    # Save to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.write_parquet(output_path)
    
    logger.info(f"✓ Merge complete!")
    logger.info(f"  Total rows: {len(combined)}")
    logger.info(f"  Date range: {combined['date'].min()} to {combined['date'].max()}")
    logger.info(f"  Unique tickers: {combined['ticker'].n_unique()}")
    logger.info(f"  Output: {output_path}")


if __name__ == "__main__":
    yfinance_dir = Path("backend/data_canonical/daily_parquet_yfinance")
    alpaca_dir = Path("backend/data_canonical/daily_parquet")
    gap_dir = Path("backend/data_canonical/daily_parquet_alpaca_gap")
    output_path = Path("backend/data/artifacts/universe/raw_ohlcv.parquet")

    merge_all_sources(yfinance_dir, alpaca_dir, output_path, gap_dir=gap_dir)
