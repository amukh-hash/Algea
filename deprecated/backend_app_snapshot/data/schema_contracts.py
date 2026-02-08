"""
Canonical Schema Contracts for Algaie Data Pipeline.
Enforces strict typing and naming conventions for all persisted artifacts.
"""

import polars as pl
from typing import List, Dict, Any

# Constants
DATE_COL = "date"
SYMBOL_COL = "symbol"

# Required Column Sets
UNIVERSEFRAME_V2_REQUIRED_COLS = [
    "date", "symbol", 
    "is_observable", "is_tradable", 
    "tier", "weight"
]

SELECTOR_FEATURES_V2_REQUIRED_COLS = [
    "date", "symbol",
    "x_lr1", "x_lr5", "x_lr20", "x_vol", "x_relvol",
    "y_rank", "tier", "weight"
]

PRIORS_REQUIRED_COLS = [
    "date", "symbol",
    "prior_drift_20d", "prior_vol_20d",
    "prior_downside_q10_20d", "prior_trend_conf_20d",
    "chronos_model_id", "context_len", "horizon", "prior_version"
]

def normalize_keys(df: pl.DataFrame) -> pl.DataFrame:
    """
    Normalize DataFrame keys to canonical schema:
    - Rename 'ticker' -> 'symbol'
    - Cast 'date' -> pl.Date
    - Cast 'symbol' -> pl.Utf8
    """
    # 1. Rename ticker -> symbol
    if "ticker" in df.columns and "symbol" not in df.columns:
        df = df.rename({"ticker": "symbol"})
        
    # 2. Cast Date
    if "date" in df.columns:
        # If datetime, cast to date
        dtype = df.schema["date"]
        if dtype != pl.Date:
            df = df.with_columns(pl.col("date").cast(pl.Date))
            
    # 3. Cast Symbol
    if "symbol" in df.columns:
        dtype = df.schema["symbol"]
        if dtype != pl.Utf8:
             df = df.with_columns(pl.col("symbol").cast(pl.Utf8))
             
    return df

def assert_schema(df: pl.DataFrame, required_cols: List[str], dtype_map: Dict[str, pl.DataType] = None) -> None:
    """
    Assert that DataFrame contains required columns and matches dtypes.
    Raises ValueError if violated.
    """
    # 1. Check Missing Columns
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Schema assertion failed. Missing columns: {missing}")
        
    # 2. Check Dtypes (if provided)
    if dtype_map:
        for col, expected_dtype in dtype_map.items():
            if col in df.columns:
                actual_dtype = df.schema[col]
                if actual_dtype != expected_dtype:
                    # Allow logical equivalence? e.g. Int32 vs Int64? 
                    # For now strict equality for critical keys.
                    raise ValueError(f"Schema assertion failed. Column '{col}' expected {expected_dtype}, got {actual_dtype}")

def schema_signature(df: pl.DataFrame) -> List[Dict[str, str]]:
    """
    Returns a JSON-serializable schema signature.
    """
    sig = []
    for name, dtype in df.schema.items():
        sig.append({"name": name, "dtype": str(dtype)})
    return sig
