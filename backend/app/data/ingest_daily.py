
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
    if not source_cfg:
        raise ValueError("source_cfg is required to ingest raw daily data.")

    parquet_path = source_cfg.get("parquet_path")
    csv_path = source_cfg.get("csv_path")
    if parquet_path:
        df = pd.read_parquet(parquet_path)
    elif csv_path:
        df = pd.read_csv(csv_path)
    else:
        raise ValueError("source_cfg must include parquet_path or csv_path.")

    if "symbol" not in df.columns and "ticker" in df.columns:
        df = df.rename(columns={"ticker": "symbol"})

    if "date" not in df.columns:
        raise ValueError("Input data must include a 'date' column.")

    df["date"] = pd.to_datetime(df["date"])

    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    return df

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
