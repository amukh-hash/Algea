
import logging
import argparse
import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta
from backend.app.ops import bootstrap, pathmap, config
from backend.app.features import featureframe

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2016-01-01")
    parser.add_argument("--end", default="2026-02-03")
    args = parser.parse_args()
    
    bootstrap.ensure_dirs()
    
    logger.info("Building FeatureFrame...")
    
    # Spec (could be loaded from file)
    spec = {
        "returns": [1, 5, 20],
        "volatility": 20,
        "volume_z": 20
    }
    
    # In a real run, we would iterate chunks of time or tickers to build the huge frame.
    # For migration PR, we assume we might migrate from `backend/data/artifacts/features/features_scaled.parquet`?
    # Check if legacy exists
    pm = pathmap.get_paths()
    legacy_path = os.path.join(pm.legacy_artifacts_root, "features/features_scaled.parquet")
    
    if os.path.exists(legacy_path) and config.ALLOW_LEGACY_READ:
        logger.info("Migrating from Legacy Features...")
        df = pd.read_parquet(legacy_path)
        
        # Map Columns to Schema B6
        # Legacy: log_return_1d, ..., volatility_20d, relative_volume_20d
        # Target: ret_1d, ..., vol_20d, volume_z_20d
        
        rename_map = {
            "log_return_1d": "ret_1d",
            "log_return_5d": "ret_5d",
            "log_return_20d": "ret_20d",
            "volatility_20d": "vol_20d",
            "relative_volume_20d": "volume_z_20d"
        }
        df.rename(columns=rename_map, inplace=True)
        
        # Add missing B6 cols
        # vol_chg_1d, dollar_vol_20d, spy_ret_1d, vix_level
        # Set dummy or defaults for migration
        if "vol_chg_1d" not in df.columns: df["vol_chg_1d"] = 0.0
        if "dollar_vol_20d" not in df.columns: df["dollar_vol_20d"] = 0.0 # Critical for filtering?
        if "spy_ret_1d" not in df.columns: df["spy_ret_1d"] = 0.0
        if "vix_level" not in df.columns: df["vix_level"] = 15.0
        
        # Write Canonical
        out_path = featureframe.write_featureframe(df, spec, version_tag="legacy_migrated")
        logger.info(f"Written FeatureFrame to {out_path}")
        
    else:
        logger.warning("No legacy features found. Full build not implemented in script 03 yet.")

if __name__ == "__main__":
    main()
