import os
import logging
import argparse
import pandas as pd
from backend.app.core import config
from backend.app.data.preproc import FeatureEngineer
from backend.app.models.scalers import SelectorScaler
from backend.app.ops import bootstrap

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv", default="backend/data/artifacts/universe/raw_ohlcv.parquet")
    parser.add_argument("--macro", default="backend/data/artifacts/features/macro_raw.parquet")
    parser.add_argument("--out_dir", default="backend/data/artifacts/features")
    parser.add_argument("--scaler_out", default="backend/data/artifacts/scalers/selector_v1.joblib")
    args = parser.parse_args()
    
    # Bootstrap
    bootstrap.ensure_dirs()
    
    # 1. Load Data
    if not os.path.exists(args.ohlcv):
        logger.error("OHLCV missing.")
        return
        
    logger.info("Loading Raw Data...")
    ohlcv = pd.read_parquet(args.ohlcv)
    
    # Macro
    macro = pd.DataFrame()
    if os.path.exists(args.macro):
        macro = pd.read_parquet(args.macro)
        # Ensure date index or column
        if 'date' not in macro.columns and isinstance(macro.index, pd.DatetimeIndex):
            macro.reset_index(inplace=True)
            macro.rename(columns={'index': 'date'}, inplace=True)
    
    # 2. Engineer Features
    logger.info("Engineering Features...")
    engineer = FeatureEngineer()
    # Mode='train' generates targets
    df_features = engineer.process_features(ohlcv, macro, mode='train')
    
    # 3. Split Train for Scaling
    # Config: TRAIN_SPLIT_DATE
    train_split = pd.Timestamp(config.TRAIN_SPLIT_DATE)
    
    train_slice = df_features[df_features['date'] < train_split]
    
    logger.info(f"Fitting Scaler on Train set (< {train_split}, {len(train_slice)} rows)...")
    if train_slice.empty:
        logger.error("Train slice empty! Check dates.")
        return
        
    scaler = SelectorScaler(method='robust')
    scaler.fit(train_slice)
    
    # Save Scaler
    os.makedirs(os.path.dirname(args.scaler_out), exist_ok=True)
    scaler.save(args.scaler_out)
    logger.info(f"Saved Scaler to {args.scaler_out}")
    
    # 4. Transform All
    logger.info("Transforming Dataset...")
    df_scaled = scaler.transform(df_features)
    
    # 5. Save Artifact
    out_path = os.path.join(args.out_dir, "features_scaled.parquet")
    os.makedirs(args.out_dir, exist_ok=True)
    df_scaled.to_parquet(out_path)
    logger.info(f"Saved Scaled Features to {out_path} ({len(df_scaled)} rows)")

if __name__ == "__main__":
    main()
