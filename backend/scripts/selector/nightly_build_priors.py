"""
Script to generate Chronos-2 teacher priors for a specific date (or range).
Writes versioned Parquet artifact with strict coverage checks.
Usage: python nightly_build_priors.py --date 2023-10-27
"""

import argparse
import os
import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
import polars as pl
import torch
import numpy as np

# Adjust path to find backend
sys.path.append(os.getcwd())

from backend.app.core import config
from backend.app.data import marketframe
from backend.app.models import chronos2_teacher
from backend.app.models.chronos2_hf_pipeline import Chronos2HFPredictor, ChronosPredictConfig
from backend.app.models.signal_types import ChronosPriors
from backend.app.ops import run_recorder

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
    parser.add_argument("--target_col", default=os.getenv("CHRONOS2_TARGET_COL", "close_adj"))
    parser.add_argument(
        "--covariate_cols",
        default=os.getenv("CHRONOS2_COVARIATE_COLS", ""),
        help="Comma-separated covariate columns for predict_df usage.",
    )

    args = parser.parse_args()

    as_of_date = datetime.strptime(args.date, "%Y-%m-%d").date()
    run_id = run_recorder.init_run(
        pipeline_type="priors",
        trigger="schedule",
        config={
            "date": args.date,
            "lookback": args.lookback,
            "horizon": args.horizon,
            "n_samples": args.n_samples,
            "min_coverage": args.min_coverage,
        },
        data_versions={"gold": "unknown", "silver": "unknown", "macro": "unknown", "universe": "unknown"},
        tags=["priors", "chronos2"],
    )
    run_dir = run_recorder.run_paths.get_run_dir(run_id)
    outputs_dir = run_dir / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Building priors for {as_of_date}")
    run_recorder.set_status(run_id, "RUNNING", stage="priors", step="build")

    # 1. Load Universe
    # Assume simple list of tickers
    # import json
    # with open(args.universe_path) as f:
    #     universe = json.load(f)
    # For now, mock universe or list files
    universe = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "SPY"] # Minimal test set

    # 2. Load Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = os.getenv("CHRONOS_MODEL_ID", "amazon/chronos-t5-tiny") # Default small for dev

    logger.info(f"Loading Chronos model: {model_id}")
    covariate_cols = [c.strip() for c in args.covariate_cols.split(",") if c.strip()]
    predictor = None
    try:
        predictor = Chronos2HFPredictor(
            ChronosPredictConfig(
                model_id=model_id,
                target_col=args.target_col,
                covariate_cols=covariate_cols,
                id_col="symbol",
                timestamp_col="timestamp",
            ),
            device=str(device),
        )
        logger.info("Loaded Chronos predict_df pipeline.")
    except Exception as exc:
        logger.warning(f"Chronos predict_df unavailable ({exc}); falling back to native wrapper.")
        predictor = None

    model = None
    if predictor is None:
        model, model_info = chronos2_teacher.load_chronos_adapter(
            model_id=model_id,
            use_qlora=False, # Teacher usually full precision or loaded as-is
            device=device,
            eval_mode=True
        )

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

            if predictor is not None:
                context_df = context.rename({"timestamp": "timestamp"}).with_columns(
                    pl.lit(ticker).alias("symbol")
                ).to_pandas()
                forecast_df = predictor.predict_df(
                    context_df=context_df,
                    prediction_length=args.horizon,
                    num_samples=args.n_samples,
                )
                priors_df = predictor.summarize_priors(forecast_df)
                row = priors_df.iloc[0]
                drift = row["teacher_drift"]
                vol = row["teacher_vol_forecast"]
                downside_q10 = row["teacher_tail_risk"]
                trend_conf = row["teacher_trend_conf"]
            else:
                target_col = args.target_col
                if target_col not in context.columns:
                    target_col = "close"
                context_values = context[target_col].to_numpy()
                context_tensor = torch.tensor(context_values, dtype=torch.float32).unsqueeze(0)
                forecast = model.generate(
                    context_tensor.unsqueeze(-1),
                    prediction_length=args.horizon,
                    num_samples=args.n_samples,
                )
                samples = forecast[0].cpu().numpy()
                last_price = context_values[-1]
                terminal_prices = samples[:, -1]
                drift = np.median(terminal_prices) / last_price - 1.0
                terminal_returns = terminal_prices / last_price - 1.0
                vol = np.std(terminal_returns)
                downside_q10 = np.quantile(terminal_returns, 0.10)
                trend_conf = np.mean(terminal_returns > 0)

            results.append({
                "date": str(as_of_date),
                "symbol": ticker,
                "teacher_drift": float(drift),
                "teacher_vol_forecast": float(vol),
                "teacher_tail_risk": float(downside_q10),
                "teacher_trend_conf": float(trend_conf),

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

    run_recorder.emit_event(
        run_id,
        stage="priors",
        step="coverage",
        level="INFO",
        message="Coverage computed",
        payload={"coverage": success_rate, "required": args.min_coverage},
    )
    if success_rate < args.min_coverage:
        logger.error(f"Coverage below threshold {args.min_coverage}. Aborting.")
        run_recorder.set_status(
            run_id,
            "FAILED",
            stage="priors",
            step="coverage",
            error={
                "type": "CoverageError",
                "message": f"Coverage {success_rate:.2%} below threshold {args.min_coverage}.",
                "traceback": "Increase universe coverage or reduce threshold.",
            },
        )
        run_recorder.finalize_run(run_id, "FAILED")
        sys.exit(1)

    # 5. Save Artifact
    df = pl.DataFrame(results)

    # Sort
    df = df.sort(["symbol"])

    # Path: backend/data/priors/chronos2/v1/YYYY-MM-DD.parquet
    out_dir = os.path.join(config.PRIORS_DIR, "chronos2", "v1")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{args.date}.parquet")

    df.write_parquet(out_path)
    logger.info(f"Saved priors to {out_path}")
    run_copy = outputs_dir / f"priors_{args.date}.parquet"
    run_copy.write_bytes(Path(out_path).read_bytes())
    run_recorder.register_artifact(
        run_id,
        name=f"priors_{args.date}",
        type="parquet",
        path=str(run_copy),
        tags=["priors"],
        meta={"date": args.date, "horizon": args.horizon},
    )
    run_recorder.finalize_run(run_id, "PASSED")

if __name__ == "__main__":
    main()
