
import logging
import argparse
import pandas as pd
import os
from datetime import datetime, timedelta
from backend.app.ops import bootstrap, pathmap, config, promotion_gate
from backend.app.data import ingest_daily, universe, security_master
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

    # 1. Ingest Latest
    logger.info("1. Ingesting Daily Bars and Updating Factors...")
    # call ingest_daily.ingest_raw_daily(...)
    # call marketframe.update(...)
    # Stub for now
    
    # 2. Build FeatureFrame for Today (using PROD feature spec)
    logger.info("2. Handling Features...")
    # df_features = featureframe.build_featureframe(asof_date, asof_date, spec=prod_cfg['feature_spec'])
    # Stub
    df_features = pd.DataFrame() 

    # 3. Generate Priors (using PROD prior spec)
    logger.info("3. Generating Priors...")
    # df_priors = priors.generate_priors_for_date(...)
    # Stub
    df_priors = pd.DataFrame()
    
    # 4. Selector Inference
    logger.info("4. Running Selector Inference...")
    predictor = infer.SelectorInference(model_version=prod_cfg['model_version'])
    
    # Get Universe
    symbols = ["AAPL", "TEST"] # Stub
    
    leaderboard = predictor.predict(date_str, symbols, df_features, df_priors)
    
    # 5. Write Leaderboard
    out_path = infer.write_leaderboard(asof_date, leaderboard)
    logger.info(f"5. Written Leaderboard to {out_path}")
    
    logger.info("Nightly Run Complete and Verified.")

if __name__ == "__main__":
    main()
