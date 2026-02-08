import os
from pathlib import Path
import logging
import argparse
import pandas as pd
from backend.app.core import config
from backend.app.data.preproc import FeatureEngineer
from backend.app.models.scalers import SelectorScaler
from backend.app.ops import bootstrap, run_recorder

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv", default="backend/data/artifacts/universe/raw_ohlcv.parquet")
    parser.add_argument("--macro", default="backend/data/artifacts/features/macro_raw.parquet")
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--scaler_out", default=None)
    args = parser.parse_args()
    
    # Bootstrap
    bootstrap.ensure_dirs()
    run_id = run_recorder.init_run(
        pipeline_type="full_pipeline",
        trigger="manual",
        config={"ohlcv": args.ohlcv, "macro": args.macro},
        data_versions={"gold": "unknown", "silver": "unknown", "macro": "unknown", "universe": "unknown"},
        tags=["preproc", "features"],
    )
    run_dir = run_recorder.run_paths.get_run_dir(run_id)
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out_dir) if args.out_dir else outputs_dir
    scaler_out = Path(args.scaler_out) if args.scaler_out else outputs_dir / "selector_scaler.joblib"
    
    # 1. Load Data
    if not os.path.exists(args.ohlcv):
        logger.error("OHLCV missing.")
        run_recorder.set_status(
            run_id,
            "FAILED",
            stage="preproc",
            step="load",
            error={"type": "FileNotFoundError", "message": "OHLCV missing.", "traceback": ""},
        )
        run_recorder.finalize_run(run_id, "FAILED")
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
    run_recorder.set_status(run_id, "RUNNING", stage="preproc", step="feature_engineering")
    df_features = engineer.process_features(ohlcv, macro, mode='train', run_id=run_id)
    
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
    os.makedirs(os.path.dirname(scaler_out), exist_ok=True)
    scaler.save(str(scaler_out))
    logger.info(f"Saved Scaler to {scaler_out}")
    run_recorder.register_artifact(
        run_id,
        name="selector_scaler",
        type="model_checkpoint",
        path=str(scaler_out),
        tags=["scaler"],
    )
    
    # 4. Transform All
    logger.info("Transforming Dataset...")
    df_scaled = scaler.transform(df_features)
    
    # 5. Save Artifact
    out_path = os.path.join(out_dir, "features_scaled.parquet")
    os.makedirs(out_dir, exist_ok=True)
    df_scaled.to_parquet(out_path)
    logger.info(f"Saved Scaled Features to {out_path} ({len(df_scaled)} rows)")
    run_recorder.register_artifact(
        run_id,
        name="features_scaled",
        type="parquet",
        path=str(out_path),
        tags=["features"],
    )
    run_recorder.finalize_run(run_id, "PASSED")

if __name__ == "__main__":
    main()
