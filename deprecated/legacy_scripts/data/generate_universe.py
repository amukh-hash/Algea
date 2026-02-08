import os
import logging
import argparse
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from backend.app.core import config
from backend.app.data.universe import UniverseSelector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def derive_ipo_dates(ohlcv_df: pd.DataFrame) -> pd.DataFrame:
    """
    Approximates IPO date as the first observed date in the dataset.
    Returns DataFrame: [ticker, ipo_date]
    """
    logger.info("Deriving IPO dates from OHLCV history...")
    g = ohlcv_df.groupby('ticker')['date'].min().reset_index()
    g.rename(columns={'date': 'ipo_date'}, inplace=True)
    return g

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv", default="backend/data/artifacts/universe/raw_ohlcv.parquet")
    parser.add_argument("--metadata", default="backend/data/artifacts/universe/master_metadata.parquet")
    parser.add_argument("--out_dir", default="backend/data/artifacts/universe")
    parser.add_argument("--start", default=config.TRAIN_START_DATE)
    parser.add_argument("--end", default=config.TRAIN_END_DATE)
    args = parser.parse_args()
    
    # 1. Load Data
    if not os.path.exists(args.ohlcv):
        logger.error(f"OHLCV artifact not found: {args.ohlcv}")
        return
        
    logger.info("Loading OHLCV data...")
    ohlcv = pd.read_parquet(args.ohlcv)
    if 'date' not in ohlcv.columns:
        # Check index
        ohlcv.reset_index(inplace=True)
        if 'timestamp' in ohlcv.columns:
            ohlcv.rename(columns={'timestamp': 'date'}, inplace=True)
    
    ohlcv['date'] = pd.to_datetime(ohlcv['date'])
    
    # 2. Load Metadata
    if os.path.exists(args.metadata):
        metadata = pd.read_parquet(args.metadata)
    else:
        logger.warning(f"Metadata not found: {args.metadata}. Creating minimal from OHLCV tickers.")
        tickers = ohlcv['ticker'].unique()
        metadata = pd.DataFrame({'ticker': tickers})
        metadata['asset_type'] = 'COMMON STOCK' # Assumption
        
    # 3. Enrich Metadata with IPO Dates
    ipos = derive_ipo_dates(ohlcv)
    if 'ipo_date' in metadata.columns:
        # Fill missing
        metadata = metadata.merge(ipos, on='ticker', how='left', suffixes=('', '_derived'))
        metadata['ipo_date'] = metadata['ipo_date'].fillna(metadata['ipo_date_derived'])
        metadata.drop(columns=['ipo_date_derived'], inplace=True)
    else:
        metadata = metadata.merge(ipos, on='ticker', how='left')

    # 4. Iterate Months
    selector = UniverseSelector()
    
    start_dt = pd.to_datetime(args.start)
    end_dt = pd.to_datetime(args.end)
    current = start_dt
    
    while current <= end_dt:
        date_str = current.strftime("%Y-%m-%d")
        logger.info(f"Generating Universe for {date_str}...")
        
        try:
            # We handle potential index errors inside selector or here (if date out of range)
            snapshot = selector.select(ohlcv, metadata, date_str)
            
            # Save only eligible? Or all with reasons?
            # Save all for audit
            out_path = os.path.join(args.out_dir, f"universe_{date_str}.parquet")
            snapshot.to_parquet(out_path)
            
            # Log stats
            eligible_count = snapshot['is_eligible'].sum()
            logger.info(f"  > Eligible: {eligible_count} / {len(snapshot)}")
            
        except Exception as e:
            logger.error(f"Failed generation for {date_str}: {e}")
        
        # Next Month
        current += relativedelta(months=1)

if __name__ == "__main__":
    main()
