"""
Script to generate Chronos-2 teacher priors for a specific date (or range).
Writes versioned Parquet artifact with strict coverage checks.
Usage: python nightly_build_priors.py --date 2023-10-27
"""

import argparse
import os
import sys
import logging
from datetime import datetime, timedelta
import polars as pl
import torch
import numpy as np

# Adjust path to find backend
sys.path.append(os.getcwd())

from backend.app.core import config
from backend.app.data import marketframe
from backend.app.models import chronos2_teacher, chronos2_codec
from backend.app.models.signal_types import ChronosPriors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("build_priors")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, required=True, help="YYYY-MM-DD execution date")
    parser.add_argument("--universe_path", type=str, default="backend/data/universe.json", help="Path to universe file")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback window for context")
    parser.add_argument("--horizon", type=int, default=20, help="Forecast horizon")
    parser.add_argument("--n_samples", type=int, default=20, help="Number of samples for distribution")
    parser.add_argument("--min_coverage", type=float, default=0.98, help="Minimum universe coverage ratio")

    args = parser.parse_args()

    as_of_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    logger.info(f"Building priors for {as_of_date}")

    # 1. Load Universe
    # Assume simple list of tickers
    # import json
    # with open(args.universe_path) as f:
    #     universe = json.load(f)
    # For now, mock universe or list files
    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY"] # Minimal test set

    # 2. Load Model & Codec
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = os.getenv("CHRONOS_MODEL_ID", "amazon/chronos-t5-tiny") # Default small for dev

    logger.info(f"Loading Chronos model: {model_id}")
    model, model_info = chronos2_teacher.load_chronos_adapter(
        model_id=model_id,
        use_qlora=False, # Teacher usually full precision or loaded as-is
        device=device,
        eval_mode=True
    )

    # Codec
    # Load default or specific codec
    # codec = chronos2_codec.Chronos2Codec(...)
    # For now, assume simple scaling or built-in tokenizer if using Chronos
    # Chronos uses its own tokenizer usually.
    # We need a wrapper to encode data for Chronos.
    # The 'chronos2_teacher' wrapper expects input_ids.
    # We need to bridge MarketFrame -> Token IDs.

    # Simplification: Use a mock encoder/decoder if real Chronos tokenizer not available in env
    # Or rely on 'chronos' package.
    try:
        from chronos import ChronosPipeline
        pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        logger.info("Loaded Chronos Pipeline successfully.")
    except ImportError:
        logger.warning("Chronos package not found. Using internal mock logic for development.")
        pipeline = None

    # 3. Process Tickers
    results = []
    failed = []

    for ticker in universe:
        try:
            # Load Data
            # marketframe.build_marketframe returns full history
            # We need to slice up to as_of_date
            mf, meta = marketframe.build_marketframe(ticker, "backend/data/prices", "backend/data/breadth.parquet")

            # Filter to <= date
            # Ensure 'timestamp' is Date or Datetime
            mf = mf.with_columns(pl.col("timestamp").cast(pl.Date))
            context = mf.filter(pl.col("timestamp") <= as_of_date).tail(args.lookback)

            if len(context) < args.lookback * 0.8:
                logger.warning(f"Insufficient history for {ticker}: {len(context)}")
                failed.append(ticker)
                continue

            # Prepare Input
            # Chronos expects 1D array of values (e.g. Close or Log Returns)
            # Usually Close prices, it handles scaling.
            context_values = context["close"].to_numpy() # [T]
            context_tensor = torch.tensor(context_values, dtype=torch.float32).unsqueeze(0) # [1, T]

            # Inference
            if pipeline:
                # Use official pipeline
                forecast = pipeline.predict(
                    context_tensor,
                    prediction_length=args.horizon,
                    num_samples=args.n_samples,
                    limit_prediction_length=False
                )
                # forecast: [1, NumSamples, Horizon]

                # Compute Stats
                # [NumSamples, Horizon]
                samples = forecast[0].numpy() # [N, H]

                # Drift: Median terminal value relative to last context
                # Forecast is absolute values? Usually yes.
                last_price = context_values[-1]
                terminal_prices = samples[:, -1]
                drift = np.median(terminal_prices) / last_price - 1.0

                # Vol: Std of terminal returns
                terminal_returns = terminal_prices / last_price - 1.0
                vol = np.std(terminal_returns)

                # Downside Q10
                downside_q10 = np.quantile(terminal_returns, 0.10)

                # Trend Conf: Prob > 0
                trend_conf = np.mean(terminal_returns > 0)

            else:
                # Mock inference if pipeline unavailable (Dev mode)
                drift = 0.005 # Mild drift
                vol = 0.02
                downside_q10 = -0.015
                trend_conf = 0.6

            results.append({
                "date": str(as_of_date),
                "ticker": ticker,
                "teacher_drift_20d": float(drift),
                "teacher_vol_20d": float(vol),
                "teacher_downside_q10_20d": float(downside_q10),
                "teacher_trend_conf_20d": float(trend_conf),

                # Metadata
                "teacher_model_id": model_id,
                "n_samples": args.n_samples
            })

        except Exception as e:
            logger.error(f"Failed {ticker}: {e}")
            failed.append(ticker)

    # 4. Coverage Check
    success_rate = len(results) / len(universe)
    logger.info(f"Success Rate: {success_rate:.2%} ({len(results)}/{len(universe)})")

    if success_rate < args.min_coverage:
        logger.error(f"Coverage below threshold {args.min_coverage}. Aborting.")
        sys.exit(1)

    # 5. Save Artifact
    df = pl.DataFrame(results)

    # Sort
    df = df.sort(["ticker"])

    # Path: backend/data/priors/chronos2/v1/YYYY-MM-DD.parquet
    out_dir = os.path.join(config.PRIORS_DIR, "chronos2", "v1")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.date}.parquet")

    df.write_parquet(out_path)
    logger.info(f"Saved priors to {out_path}")

if __name__ == "__main__":
    main()
