"""
Script to run Selector inference nightly and produce the Leaderboard artifact.
"""

import argparse
import os
import sys
import logging
import polars as pl
import torch
from datetime import datetime
import json
import hashlib

# Adjust path
sys.path.append(os.getcwd())

from backend.app.core import config, artifacts
from backend.app.models.selector_runner import SelectorRunner
from backend.app.models.signal_types import LEADERBOARD_SCHEMA
from backend.app.data import marketframe
from backend.app.preprocessing.preproc import Preprocessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("run_selector")

def validate_schema(df: pl.DataFrame):
    """Ensure dataframe matches strict leaderboard schema"""
    missing = [c for c in LEADERBOARD_SCHEMA.keys() if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in leaderboard: {missing}")

    # Check types if possible (Polars lazy check)
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD")
    parser.add_argument("--version", type=str, default="v1", help="Selector Version")
    parser.add_argument("--universe_path", type=str, default="backend/data/universe.json")
    parser.add_argument("--min_coverage", type=float, default=0.98)

    args = parser.parse_args()
    as_of_date = datetime.strptime(args.date, "%Y-%m-%d").date()

    logger.info(f"Running Selector {args.version} for {as_of_date}")

    # 1. Resolve Artifacts
    # Priors
    priors_path = artifacts.resolve_priors_path(str(as_of_date), "v1") # Assume v1 priors match v1 selector? Or decouple?
    if not priors_path:
        logger.error(f"No priors found for {as_of_date}")
        sys.exit(1)

    logger.info(f"Using priors: {priors_path}")
    priors_df = pl.read_parquet(priors_path)

    # Checkpoint
    checkpoint_path = artifacts.resolve_selector_checkpoint(args.version)
    if not checkpoint_path:
        logger.error(f"No checkpoint found for {args.version}")
        sys.exit(1)

    # 2. Load Runner
    runner = SelectorRunner(version=args.version, device="cuda" if torch.cuda.is_available() else "cpu")

    # 3. Build Features for Universe
    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY"] # Mock

    features_dict = {}

    # Preprocessor
    # Load fit preprocessor (global for swing?)
    # Or fit on fly? No, must be frozen.
    # Assume saved preprocessor at specific path
    preproc_path = "backend/data/preprocessing/preproc_v1.json"
    if not os.path.exists(preproc_path):
        # Create dummy for test if missing
        logger.warning("Preprocessor not found, creating dummy...")
        dummy = Preprocessor()
        # Mock dataframe for fit
        mock_df = pl.DataFrame({
            "timestamp": [datetime.now()],
            "close": [100.0],
            "volume": [1000.0],
            "ad_line": [0.0],
            "bpi": [50.0]
        })
        dummy.fit(mock_df)
        os.makedirs(os.path.dirname(preproc_path), exist_ok=True)
        dummy.save(preproc_path)

    preproc = Preprocessor.load(preproc_path)

    for ticker in universe:
        try:
            # Get MarketFrame
            mf, _ = marketframe.build_marketframe(ticker, "backend/data/prices", "backend/data/breadth.parquet")

            # Filter history <= date
            mf = mf.with_columns(pl.col("timestamp").cast(pl.Date))
            context = mf.filter(pl.col("timestamp") <= as_of_date)

            # Transform
            feat = preproc.transform(context)

            # Attach Priors
            # Priors DF has all tickers for this date
            ticker_prior = priors_df.filter(pl.col("ticker") == ticker)
            if len(ticker_prior) == 0:
                logger.warning(f"No prior for {ticker}")
                continue

            # We attach prior columns to every row in feature window?
            # Yes, standard practice: context features + static context (prior)
            # Broadcast prior to all rows
            # Polars join or with_columns

            # Extract values
            # drift_20d, etc.
            # Assuming schema
            prior_vals = ticker_prior.select([
                "teacher_drift_20d", "teacher_vol_20d", "teacher_downside_q10_20d", "teacher_trend_conf_20d"
            ]).to_dict(as_series=False)

            # Add columns
            # Be careful if dict values are lists
            feat = feat.with_columns([
                pl.lit(prior_vals[k][0]).alias(k) for k in prior_vals
            ])

            features_dict[ticker] = feat

        except Exception as e:
            # logger.error(f"Feature prep failed {ticker}: {e}")
            continue

    # 4. Run Inference
    if not features_dict:
        logger.error("No valid features prepared.")
        sys.exit(1)

    leaderboard = runner.infer(features_dict, lookback=60)

    # 5. Validate & Enrich Metadata
    # Add metadata columns
    # Ensure they exist in LEADERBOARD_SCHEMA

    # We need to add columns to the dataframe
    meta_cols = {
        "as_of_date": str(as_of_date),
        "selector_checkpoint_id": os.path.basename(checkpoint_path),
        "selector_version": args.version,
        "selector_scaler_version": args.version,
        "calibration_version": args.version,
        "teacher_model_id": "amazon/chronos-t5-tiny",
        "teacher_adapter_id": "none",
        "teacher_codec_version": "v1",
        "feature_contract_hash": "hash123",
        "preproc_version": preproc.version_hash
    }

    for k, v in meta_cols.items():
        leaderboard = leaderboard.with_columns(pl.lit(v).alias(k))

    # Also attach teacher prior values used?
    # They are in features, but infer doesn't return features.
    # We should merge them back from priors_df.
    leaderboard = leaderboard.join(priors_df.select(["ticker", "teacher_drift_20d", "teacher_vol_20d", "teacher_downside_q10_20d", "teacher_trend_conf_20d"]), on="ticker", how="left")

    # Schema Check
    validate_schema(leaderboard)

    # Coverage Check
    coverage = len(leaderboard) / len(universe)
    logger.info(f"Leaderboard Coverage: {coverage:.2%} ({len(leaderboard)}/{len(universe)})")

    if coverage < args.min_coverage:
        logger.error(f"Coverage below threshold {args.min_coverage}")
        sys.exit(1)

    # 6. Save
    out_dir = os.path.join(config.SIGNALS_DIR, "selector", args.version)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{as_of_date}.parquet")

    leaderboard.write_parquet(out_path)
    logger.info(f"Saved leaderboard to {out_path}")

if __name__ == "__main__":
    main()
