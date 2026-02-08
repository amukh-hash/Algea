
import polars as pl
import pandas as pd
from pathlib import Path
from typing import Optional, List, Union
import logging
from backend.app.ops import pathmap
from backend.app.data.schema_contracts import normalize_keys

logger = logging.getLogger(__name__)

# Lightweight LRU Cache for UniverseFrame
_UNIVERSE_CACHE = {}

def load_universe_frame(
    start_date: Optional[str] = None, 
    end_date: Optional[str] = None, 
    columns: Optional[List[str]] = None,
    version: str = "v2"
) -> pl.DataFrame:
    """
    Load UniverseFrame (V2) from canonical location or specified root.
    Handles Hive partitioning (recursive scan) and key normalization.
    """
    # Resolve path
    base_path = pathmap.get_universe_frame_root(version=version)
    
    if not base_path.exists():
        logger.warning(f"UniverseFrame not found at {base_path}")
        return pl.DataFrame()

    # Scan Parquet (Recursive)
    # Hive partitioning often implies recursive structure
    if base_path.is_file():
         scan_path = str(base_path)
    else:
         scan_path = str(base_path / "**/*.parquet")
    
    try:
        lf = pl.scan_parquet(scan_path, hive_partitioning=True)
    except Exception as e:
        logger.error(f"Failed to scan universe frame: {e}")
        return pl.DataFrame()

    # Normalize Keys (Defensive)
    # We do this after collect usually, but let's try to do it lazily if possible
    # Rename ticker -> symbol if present
    
    # Filter Date (Pushdown)
    if start_date:
        d = pl.lit(start_date).cast(pl.Date)
        lf = lf.filter(pl.col("date") >= d)
    if end_date:
        d = pl.lit(end_date).cast(pl.Date)
        lf = lf.filter(pl.col("date") <= d)
        
    # Select Columns (Pushdown)
    if columns:
        # Map requested columns if they used "ticker"
        columns = ["symbol" if c == "ticker" else c for c in columns]
        
        # Ensure mandatory index cols present
        reqs = ["date", "symbol"]
        # If schema has 'ticker' instead of 'symbol' (legacy), we need to handle that
        # But we can't easily know without collecting schema.
        # Let's collect schema first?
        # Or just select all then shrink after collect?
        # Safer to select all then shrink, unless huge.
        pass # defer selection to after normalize
        
    # Collect
    df = lf.collect()
    
    # Normalize
    df = normalize_keys(df)
    
    # Final Projection
    if columns:
        cols_to_select = list(set(["date", "symbol"] + columns))
        # intersection with available
        available = set(df.columns)
        final = [c for c in cols_to_select if c in available]
        df = df.select(final)
        
    return df

def get_universe_mask(date: str, kind: str = "tradable") -> pd.DataFrame:
    """
    Get mask for a specific date. 
    Returns pandas DataFrame indexed by symbol with boolean column 'mask'.
    Kind: 'tradable' or 'observable'.
    """
    col = "is_tradable" if kind == "tradable" else "is_observable"
    
    df = load_universe_frame(start_date=date, end_date=date, columns=[col])
    
    if df.height == 0:
        return pd.DataFrame()
        
    # Convert to pandas 
    pdf = df.to_pandas()
    # Use symbol as index
    if "symbol" in pdf.columns:
        pdf.set_index("symbol", inplace=True)
    elif "ticker" in pdf.columns: # Should not happen with new loader
        pdf.set_index("ticker", inplace=True)
        
    return pdf[[col]]

def get_weights(date: str) -> pd.DataFrame:
    """
    Get weights for a specific date.
    Returns pandas DataFrame indexed by symbol with 'weight' column.
    """
    df = load_universe_frame(start_date=date, end_date=date, columns=["weight", "is_tradable"])
    
    if df.height == 0:
        return pd.DataFrame()
        
    pdf = df.to_pandas()
    # Use symbol as index
    if "symbol" in pdf.columns:
        pdf.set_index("symbol", inplace=True)
    return pdf[["weight"]]

def get_tradable_tickers(date: str) -> List[str]:
    """
    Get list of tradable symbols for a date.
    """
    df = load_universe_frame(start_date=date, end_date=date, columns=["is_tradable"])
    if df.height == 0:
        return []
    
    return df.filter(pl.col("is_tradable")).select("symbol").to_series().to_list()
