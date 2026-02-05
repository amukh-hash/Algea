
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
    # symbols = list(feature_spec.get('symbols', ['AAPL']))
    # Start with mock MarketFrame
    df = marketframe.build_marketframe(start_date, end_date, symbols=['AAPL', 'MSFT'])
    
    # 2. Compute Features (Mock Logic matching B6 Schema)
    df['ret_1d'] = 0.001
    df['ret_5d'] = 0.005
    df['ret_20d'] = 0.02
    df['vol_20d'] = 0.015
    df['vol_chg_1d'] = 0.0
    df['dollar_vol_20d'] = 1e6
    df['volume_z_20d'] = 0.5
    
    # Covariates
    df['spy_ret_1d'] = 0.001
    df['vix_level'] = 20.0
    
    # Metadata columns
    df['feature_version'] = "v1"
    df['data_version'] = "v1"
    
    # Ensure columns match B6 schema
    required = schemas.SCHEMA_FEATUREFRAME["columns"].keys()
    # Filter or fill missing (done by explicit assignment above)
    
    # Return B6 DataFrame
    return df[list(required)].copy()

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
