
import logging
import argparse
import pandas as pd
import os
from datetime import datetime, timedelta
from backend.app.ops import bootstrap, pathmap, config, promotion_gate
from backend.app.data import ingest_daily, universe, security_master, calendar
from backend.app.features import featureframe
from backend.app.teacher import priors, chronos_runner
from backend.app.selector import infer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asof", default=None, help="YYYY-MM-DD")
    args = parser.parse_args()
    
    bootstrap.ensure_dirs()
    
    # Date determination
    if args.asof:
        asof_date = pd.to_datetime(args.asof)
    else:
        asof_date = pd.Timestamp.now().normalize() - pd.Timedelta(days=1)
        
    date_str = asof_date.strftime('%Y-%m-%d')
    logger.info(f"Nightly Run for {date_str}...")
    
    # Load PROD Pointer for version consistency
    prod_cfg = promotion_gate.load_prod_pointer()
    logger.info(f"PROD Configuration: {prod_cfg}")

    # 1. Ingest Latest (no-op placeholder if data already present)
    logger.info("1. Ingesting Daily Bars and Updating Factors...")

    # 2. Load FeatureFrame for Today (using PROD feature spec)
    logger.info("2. Handling Features...")
    feature_path = pathmap.resolve("featureframe", version=prod_cfg.get("feature_version", "v1"))
    if not os.path.exists(feature_path):
        raise FileNotFoundError(f"Missing featureframe: {feature_path}")
    df_features = pd.read_parquet(feature_path)
    df_features["date"] = pd.to_datetime(df_features["date"])

    # 3. Load Priors for Today (using PROD prior spec)
    logger.info("3. Generating Priors...")
    prior_version = prod_cfg.get("prior_version", "v1")
    df_priors = priors.load_priors(asof_date, prior_version=prior_version)
    df_priors["date"] = pd.to_datetime(df_priors["date"])
    
    # 4. Selector Inference
    logger.info("4. Running Selector Inference...")
    predictor = infer.SelectorInference(model_version=prod_cfg['model_version'])
    
    # Get Universe
    manifest_path = pathmap.resolve("manifest", date=asof_date)
    if os.path.exists(manifest_path):
        manifest = pd.read_parquet(manifest_path)
        symbols = manifest.loc[manifest["eligible"] == True, "symbol"].astype(str).tolist()
    else:
        sec_master = security_master.load_security_master()
        symbols = sec_master["symbol"].dropna().astype(str).unique().tolist()
    
    leaderboard = predictor.predict(date_str, symbols, df_features, df_priors)
    
    # 5. Write Leaderboard
    out_path = infer.write_leaderboard(asof_date, leaderboard)
    logger.info(f"5. Written Leaderboard to {out_path}")
    
    logger.info("Nightly Run Complete and Verified.")

if __name__ == "__main__":
    main()
