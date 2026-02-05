
import pandas as pd
import numpy as np
import logging
from typing import List, Dict
from backend.app.ops import pathmap, config, artifact_registry
from backend.app.features import schemas, validators
from backend.app.data import security_master

logger = logging.getLogger(__name__)

def build_universe_manifest(asof_date, base_symbols: List[str], rules: Dict) -> pd.DataFrame:
    """
    Applies filtering rules to base symbols for a given date.
    Returns Schema B5 DataFrame.
    """
    # 1. Gather Data for filtering (Price, ADV, IPO Date, Sector)
    # This implies loading daily bars for all symbols? Expensive?
    # Or assuming base_symbols contains pre-computed metrics?
    # Ideally: load metrics from a "MarketFrame" or similar cache.
    # For bootstrap, we might have to inefficiently load OHLCV partitions.
    # Or rely on "02_build_universe" script to pass in enriched data.
    
    # We'll assume the caller (script) prepares a DataFrame with columns:
    # [symbol, close_adj, adv20, vol20, ipo_date, sector]
    
    # FOR NOW: Stub the data gathering inside script, here we implement logic on DF.
    pass

def apply_universe_rules(metrics_df: pd.DataFrame, rules: Dict) -> pd.DataFrame:
    """
    metrics_df expected cols: symbol, asset_type, close_adj, adv20, vol20, ipo_date, sector
    """
    df = metrics_df.copy()
    
    # Defaults
    min_price = rules.get("min_price", 5.0)
    min_adv = rules.get("min_adv", 25e6)
    min_ipo_days = rules.get("min_ipo_days", 252)
    top_n = rules.get("top_n", config.UNIVERSE_BUFFER_MAX)
    
    # Reason Codes
    df["eligible"] = False
    df["reason_code"] = "OK"
    
    # 1. Asset Type
    mask_common = df["asset_type"] == "COMMON"
    df.loc[~mask_common, "reason_code"] = "NON_COMMON"
    
    # 2. Price
    mask_price = df["close_adj"] >= min_price
    # Only overwrite if currently OK
    mask_ok = df["reason_code"] == "OK"
    df.loc[mask_ok & ~mask_price, "reason_code"] = "PRICE_LT_5"
    
    # 3. IPO Age
    # df['ipo_age_td'] should be provided or calc from ipo_date
    # Stub logic for now
    if "ipo_age_td" not in df.columns:
        df["ipo_age_td"] = 9999
    
    mask_ok = df["reason_code"] == "OK"
    mask_ipo = df["ipo_age_td"] >= min_ipo_days
    df.loc[mask_ok & ~mask_ipo, "reason_code"] = "IPO_LT_252TD"
    
    # 4. Liquidity
    mask_ok = df["reason_code"] == "OK"
    mask_adv = df["adv20"] >= min_adv
    df.loc[mask_ok & ~mask_adv, "reason_code"] = "ADV20_LT_25M"
    
    # 5. Sector Cap? (Optional)
    
    # Final Eligibility
    df.loc[df["reason_code"] == "OK", "eligible"] = True
    
    # Top N Filtering (if too many eligible)
    eligible = df[df["eligible"]].copy()
    if len(eligible) > top_n:
        # Sort by ADV descending
        eligible = eligible.sort_values("adv20", ascending=False)
        cutoff = eligible.iloc[top_n]["adv20"]
        # Mark those below cutoff
        mask_cut = (df["eligible"]) & (df["adv20"] < cutoff)
        df.loc[mask_cut, "eligible"] = False
        df.loc[mask_cut, "reason_code"] = "ADV_RANK_CUTOFF"
        
    return df

def write_universe_manifest(asof_date, df: pd.DataFrame, rules: Dict) -> str:
    path = pathmap.resolve("manifest", date=asof_date)
    
    # Versioning
    code_version = "v1" # Git hash ideally
    uv = artifact_registry.compute_universe_version(rules, code_version)
    df["universe_version"] = uv
    df["asof_date"] = pd.to_datetime(asof_date)
    
    # Renaming to Schema B5
    # Input has 'adv20' -> Schema 'adv20_median'
    rename_map = {"adv20": "adv20_median", "vol20": "vol20_median"}
    df.rename(columns=rename_map, inplace=True)
    
    # Validate
    validators.validate_df(df, schemas.SCHEMA_UNIVERSE_MANIFEST, context="Universe Write")
    
    # Ensure dir
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_parquet(path)
    
    return path
