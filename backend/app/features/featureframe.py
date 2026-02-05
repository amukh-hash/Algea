
import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict
from backend.app.ops import pathmap, config, artifact_registry
from backend.app.features import schemas, validators
from backend.app.data import marketframe

logger = logging.getLogger(__name__)

def build_featureframe(
    start_date, end_date, feature_spec: Dict
) -> pd.DataFrame:
    """
    Computes features for the given window.
    Returns Schema B6.
    """
    # 1. Load Data via MarketFrame (or iterate tickers)
    # For efficiency in this script, we likely iterate tickers, compute features, and stack.
    
    # Mocking the loop for the template:
    # symbols = ...
    
    # 2. Compute Features
    # Feature Engineering Logic (moved from preproc.py)
    # Needs:
    # - ret_1d, ret_5d, ret_20d
    # - vol_20d
    # - vol_chg_1d
    # - dollar_vol_20d
    # - volume_z_20d
    
    # df['log_ret'] = np.log(df['close_adj'] / df['close_adj'].shift(1))
    # ...
    
    # 3. Validate
    # validators.validate_df(df, schemas.SCHEMA_FEATUREFRAME)
    
    return pd.DataFrame()

def write_featureframe(df: pd.DataFrame, feature_spec: Dict, version_tag: str = "v1") -> str:
    # Compute Version
    code_version = "v1"
    fv = artifact_registry.compute_feature_version(feature_spec, code_version)
    
    df["feature_version"] = fv
    df["data_version"] = "v1" # From config?
    
    path = pathmap.resolve("featureframe", version=version_tag)
    
    # Validate before write
    validators.validate_df(df, schemas.SCHEMA_FEATUREFRAME, context="FeatureFrame Write")
    
    # Write
    df.to_parquet(path)
    
    # Metadata
    meta = {
        "version": fv,
        "spec": feature_spec,
        "rows": len(df),
        "columns": list(df.columns)
    }
    artifact_registry.write_metadata(path + ".meta", meta)
    
    return path
