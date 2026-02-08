
import pandas as pd
import logging
from typing import Dict, Any, Optional
from backend.app.ops import pathmap, config, artifact_registry
from backend.app.features import schemas, validators

logger = logging.getLogger(__name__)

def build_security_master(raw_data: pd.DataFrame, data_version: str) -> pd.DataFrame:
    """
    Constructs the canonical security master from raw inputs.
    raw_data: Expected minimal cols [ticker, asset_type, ...]. 
              If from Alpaca, might have [symbol, name, exchange, ...].
    """
    # 1. Map columns to Schema B1
    # Schema: symbol, primary_id, exchange, asset_type, ipo_date, delist_date, sector, industry
    
    df = raw_data.copy()
    
    # Normalization
    if "symbol" not in df.columns and "ticker" in df.columns:
        df.rename(columns={"ticker": "symbol"}, inplace=True)
        
    df["symbol"] = df["symbol"].astype("string")
    
    # Primary ID: If missing, use symbol (strict requirement)
    if "primary_id" not in df.columns:
        df["primary_id"] = df["symbol"]
    
    # Asset Type
    # If raw has 'class' or 'type', map to asset_type
    # Alpaca has 'class', 'exchange'
    if "asset_type" not in df.columns:
        df["asset_type"] = "COMMON" # Default fallback
        
    # Validation against B1
    # Ensure types
    for col in ["exchange", "sector", "industry"]:
        if col not in df.columns:
            df[col] = None
        df[col] = df[col].astype("string")
            
    df["ipo_date"] = pd.to_datetime(df.get("ipo_date", "1900-01-01"))
    df["delist_date"] = pd.to_datetime(df.get("delist_date", pd.NaT))
    
    # Filter to Schema columns only
    cols = list(schemas.SCHEMA_SECURITY_MASTER["columns"].keys())
    master = df[cols].copy()
    
    # validate
    validators.validate_df(master, schemas.SCHEMA_SECURITY_MASTER, context="Security Master Build")
    validators.enforce_unique(master, ["symbol"])
    
    return master

def write_security_master(df: pd.DataFrame, version: str) -> str:
    paths = pathmap.get_paths()
    out_path = pathmap.resolve("security_master") # Canonical path
    
    # Write Parquet
    df.to_parquet(out_path)
    
    # Sidecar
    metadata = {
        "rows": len(df),
        "version": version,
        "schema": "B1",
        "hash": artifact_registry.stable_hash({"v": version, "len": len(df)})
    }
    artifact_registry.write_metadata(out_path + ".meta", metadata)
    
    return out_path
    
def load_security_master() -> pd.DataFrame:
    path = pathmap.resolve("security_master")
    return pd.read_parquet(path)

def is_common_equity(symbol: str, asof_date=None) -> bool:
    # This implies loading the master every call? Or cached?
    # Ideally cached singleton.
    # For now, load once per process (lru_cache) if performance issues.
    # Stub for now
    return True

def get_sector(symbol: str) -> Optional[str]:
    # Stub
    return None
