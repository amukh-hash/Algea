
import pandas as pd
import os
import logging
from backend.app.ops import pathmap, config, artifact_registry
from backend.app.features import schemas, validators

logger = logging.getLogger(__name__)

def ingest_raw_daily(source_cfg: dict, start_date, end_date) -> pd.DataFrame:
    """
    Ingests raw data from source (e.g. Alpaca).
    Returns DataFrame matching minimal raw schema.
    """
    # Logic to fetch from Alpaca (similar to fetch_data.py but return DF)
    # For now, we assume fetch_data.py does the heavy lifting or we call a provider class
    # Stub for migration - returning valid mock data structure to unblock pipeline
    dates = pd.date_range(start=start_date, end=end_date, freq='B')
    data = []
    # Mock for a few symbols
    for s in ['AAPL', 'MSFT', 'GOOGL']:
        for d in dates:
            data.append({
                'symbol': s,
                'date': d,
                'open': 150.0,
                'high': 155.0,
                'low': 149.0,
                'close': 152.0,
                'volume': 1000000,
                'vwap': 151.0
            })
    return pd.DataFrame(data)

def write_ohlcv_partition(symbol: str, df: pd.DataFrame) -> None:
    """
    Writes partitioned parquet for a single ticker.
    Path: backend/data_canonical/ohlcv_adj/ticker=XYZ/data.parquet
    """
    paths = pathmap.get_paths()
    base_dir = os.path.join(paths.data_canonical, "ohlcv_adj", f"ticker={symbol}")
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, "data.parquet")
    
    # Validation
    validators.validate_df(df, schemas.SCHEMA_OHLCV_ADJ, context=f"OHLCV {symbol}")
    
    df.to_parquet(out_path)
    
def load_ohlcv(symbol: str, start_date=None, end_date=None) -> pd.DataFrame:
    paths = pathmap.get_paths()
    path = os.path.join(paths.data_canonical, "ohlcv_adj", f"ticker={symbol}", "data.parquet")
    
    if not os.path.exists(path):
        # Fallback needed if ALLOW_LEGACY_READ?
        # Or just return empty
        return pd.DataFrame()
        
    df = pd.read_parquet(path)
    # Filter dates
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]
        
    return df
