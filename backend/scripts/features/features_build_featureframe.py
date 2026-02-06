
import logging
import argparse
import pandas as pd
import numpy as np
import os
from dateutil.relativedelta import relativedelta
from backend.app.ops import bootstrap, pathmap, config
from backend.app.features import build_featureframe

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
        "returns": [1, 3, 5, 10],
        "volatility": 20,
        "volume_z": 20
    }
    
    df = build_featureframe.build_featureframe(args.start, args.end, spec)
    out_path = build_featureframe.write_featureframe(df, spec, version_tag="v1")
    logger.info(f"Written FeatureFrame to {out_path}")

if __name__ == "__main__":
    main()
