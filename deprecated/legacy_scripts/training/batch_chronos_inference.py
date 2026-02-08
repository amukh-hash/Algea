import os
import logging
import argparse
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datetime import timedelta
from backend.app.core import config, artifacts
from backend.app.models.chronos2_teacher import load_chronos_adapter, infer_priors
from backend.app.models.chronos2_hf_pipeline import Chronos2HFPredictor, ChronosPredictConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ohlcv", default="backend/data/artifacts/universe/raw_ohlcv.parquet")
    parser.add_argument("--out_dir", default="backend/data/artifacts/priors")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--model_id", default="amazon/chronos-t5-small") # User can override
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lookback", type=int, default=512)
    parser.add_argument("--step_days", type=int, default=1) # Run every day? Or weekly? Plan implies daily.
    parser.add_argument("--target_col", default=os.getenv("CHRONOS2_TARGET_COL", "close"))
    parser.add_argument(
        "--covariate_cols",
        default=os.getenv("CHRONOS2_COVARIATE_COLS", ""),
        help="Comma-separated covariate columns for predict_df usage.",
    )
    args = parser.parse_args()

    # 1. Load Data
    logger.info("Loading OHLCV...")
    df = pd.read_parquet(args.ohlcv)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['ticker', 'date'])
    
    # 2. Load Model
    logger.info(f"Loading Model {args.model_id} on {args.device}...")
    covariate_cols = [c.strip() for c in args.covariate_cols.split(",") if c.strip()]
    predictor = None
    try:
        predictor = Chronos2HFPredictor(
            ChronosPredictConfig(
                model_id=args.model_id,
                target_col=args.target_col,
                covariate_cols=covariate_cols,
                id_col="ticker",
                timestamp_col="date",
            ),
            device=args.device,
        )
        logger.info("Using Chronos predict_df pipeline.")
    except Exception as exc:
        logger.warning(f"predict_df unavailable ({exc}); falling back to native wrapper.")

    if predictor is not None and args.target_col not in df.columns:
        logger.warning(f"Target column '{args.target_col}' missing; falling back to native wrapper.")
        predictor = None

    wrapper = None
    if predictor is None:
        wrapper, model_info = load_chronos_adapter(
            args.model_id,
            use_qlora=False,
            device=torch.device(args.device),
            eval_mode=True,
        )
    
    # 3. Simulate History
    start_dt = pd.to_datetime(config.TRAIN_START_DATE)
    end_dt = pd.to_datetime(config.TRAIN_END_DATE)
    current = start_dt
    
    # Pre-compute unique dates to iterate faster?
    all_dates = df['date'].unique()
    all_dates.sort()
    
    # Filter for training window
    target_dates = [d for d in all_dates if d >= start_dt and d <= end_dt]
    
    # We can step by N days
    target_dates = target_dates[::args.step_days]
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    logger.info(f"Starting Batch Inference for {len(target_dates)} dates...")
    
    for date in tqdm(target_dates):
        date_str = pd.Timestamp(date).strftime("%Y-%m-%d")
        out_path = os.path.join(args.out_dir, f"priors_{date_str}.parquet")
        
        if os.path.exists(out_path):
            continue # Resume support
            
        # Extract Sequences for this date
        # We need (Ticker, Last 512 Close Prices)
        # Filter: date <= current_date
        # Efficient way: df[df.date <= date].groupby('ticker').tail(lookback)
        # But this is slow inside loop.
        # Better: Maintain a rolling buffer? 
        # Or just trust Polars execution if we had it.
        # With Pandas:
        
        # Optimization: subset around date
        # lookback_start = date - 512 days * 2 (approx buffer)
        # causal_slice = df[(df.date <= date) & (df.date > date - timedelta(days=1000))]
        # But trading days != calendar days.
        
        # Let's use the robust groupby logic, optimization later if slow.
        # We need EXACTLY 512 points ending ON 'date'.
        
        # Trick: df indexed by (ticker, date).
        # We can shift? 
        
        # Naive approach for correctness first:
        subset = df[df['date'] <= date].groupby('ticker').tail(args.lookback)
        
        # Check lengths
        counts = subset.groupby('ticker').size()
        valid_tickers = counts[counts == args.lookback].index
        
        if len(valid_tickers) == 0:
            continue
            
        valid_subset = subset[subset['ticker'].isin(valid_tickers)].copy()
        
        priors_batch = []

        if predictor is not None:
            context_df = valid_subset.copy()
            context_df["ticker"] = context_df["ticker"].astype(str)
            forecast_df = predictor.predict_df(
                context_df=context_df,
                prediction_length=10,
                num_samples=20,
            )
            priors_df = predictor.summarize_priors(forecast_df)
            for _, row in priors_df.iterrows():
                priors_batch.append(
                    {
                        "date": date,
                        "ticker": row["ticker"],
                        "drift": row["teacher_drift"],
                        "vol_forecast": row["teacher_vol_forecast"],
                        "tail_risk": row["teacher_tail_risk"],
                        "trend_conf": row["teacher_trend_conf"],
                    }
                )
        else:
            # [N * 512]
            values = valid_subset[args.target_col].values.reshape(len(valid_tickers), args.lookback)
            tickers = valid_tickers.tolist()
            input_tensor = torch.tensor(values, dtype=torch.float32).unsqueeze(-1).to(args.device)

            for i in range(0, len(tickers), args.batch_size):
                batch_in = input_tensor[i : i+args.batch_size]
                batch_tickers = tickers[i : i+args.batch_size]

                try:
                    batch_priors = infer_priors(wrapper, batch_in, horizon_days=10, n_samples=20)
                    for t, p in zip(batch_tickers, batch_priors):
                        priors_batch.append({
                            "date": date,
                            "ticker": t,
                            "drift": p.drift,
                            "vol_forecast": p.vol_forecast,
                            "tail_risk": p.tail_risk,
                            "trend_conf": p.trend_conf
                        })
                except Exception as e:
                    logger.error(f"Inference failed for batch {i}: {e}")
                
        # Save
        if priors_batch:
            res_df = pd.DataFrame(priors_batch)
            res_df.to_parquet(out_path)

if __name__ == "__main__":
    main()
