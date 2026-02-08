
import os
import sys
import argparse
import polars as pl
import pandas as pd
import numpy as np
import torch
import json
import logging
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta

# Ensure backend in path
sys.path.append(os.getcwd())

from backend.app.ops import bootstrap, run_recorder, pathmap, config
from backend.app.models.chronos2_teacher import load_chronos_adapter, Chronos2NativeWrapper
from backend.app.data import priors_store
from backend.app.data.ingest import ohlcv_daily

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_priors_batch(model_wrapper: Chronos2NativeWrapper, context_tensor: torch.Tensor, device: str) -> dict:
    """
    Run inference and extract stats.
    Input: [B, ContextLen]
    Output: Dict of tensors {mu5, sig5, ...}
    """
    # Predict Quantiles
    # Chronos native predict returns [B, NumSamples, Horizon] or [B, Quantiles, Horizon] depending on mode.
    # We use predict_quantiles directly if available for efficiency?
    # Or pipeline.predict? pipeline is slower.
    # Chronos2NativeWrapper should expose efficient predict.
    
    # Let's use internal generate?
    # Or predict_quantiles if implemented.
    # If not, use pipeline-style generation (sample paths).
    
    # We want robust stats, efficiently.
    # If we use `predict` with num_samples=20, we get [B, 20, H].
    # Then compute stats.
    
    forecast_horizon = config.TEACHER_FORECAST_HORIZON_DAYS # 30
    
    with torch.no_grad():
        # context_tensor: [B, T]
        # output: [B, NumSamples, Horizon]
        # We need to ensure context is tensor.
        forecasts = model_wrapper.generate(
            context_tensor,
            prediction_length=forecast_horizon,
            num_samples=20,
        )
        
    # Forecasts: [B, S, H]
    # Calculate cumulative returns for 5d and 10d
    # Log returns are additive.
    # H=30.
    
    # 5D: sum(0..4)
    # 10D: sum(0..9)
    
    f_5d = forecasts[:, :, :5].sum(dim=2) # [B, S]
    f_10d = forecasts[:, :, :10].sum(dim=2) # [B, S]
    
    # Stats
    # Mu
    mu5 = f_5d.mean(dim=1)
    mu10 = f_10d.mean(dim=1)
    
    # Sigma
    sig5 = f_5d.std(dim=1)
    sig10 = f_10d.std(dim=1)
    
    # Prob Down (< 0)
    p_down5 = (f_5d < 0).float().mean(dim=1)
    p_down10 = (f_10d < 0).float().mean(dim=1)
    
    return {
        "p_mu5": mu5.cpu().numpy(),
        "p_mu10": mu10.cpu().numpy(),
        "p_sig5": sig5.cpu().numpy(),
        "p_sig10": sig10.cpu().numpy(),
        "p_pdown5": p_down5.cpu().numpy(),
        "p_pdown10": p_down10.cpu().numpy()
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start_date", default="2023-01-01")
    parser.add_argument("--end_date", default="2023-01-31")
    parser.add_argument("--ticker", help="Specific ticker or None for all in universe")
    parser.add_argument("--adapter_path", help="Path to checkpoint adapter")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--out_dir", default="backend/data/priors/chronos2/v1")
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    
    bootstrap.ensure_dirs()
    
    run_id = run_recorder.init_run(
        pipeline_type="generate_priors",
        trigger="manual",
        config=vars(args),
        data_versions={}, # Todo
        tags=["priors", "chronos2"]
    )
    
    try:
        # 1. Load Model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Chronos-2 adapter from {args.adapter_path} on {device}...")
        
        # Load wrapper
        wrapper, info = load_chronos_adapter(
            model_id="amazon/chronos-2",
            use_qlora=False,
            device=device,
            adapter_path=args.adapter_path,
            eval_mode=True
            # use_qlora? inference usually fp16 or bf16. 
            # assume wrapper handles defaults.
        )
        wrapper.eval()
        
        # 2. Determine Universe/Dates
        # Load Universe Frame to get valid tickers per date?
        # Or just raw OHLCV?
        # We should only generate for "Tradeable" or "Observable"?
        # Let's iterate all tickers found in daily_parquet for simplicity & coverage.
        
        # Get list of tickers
        daily_root = "backend/data_canonical/daily_parquet"
        if args.ticker:
            tickers = [args.ticker]
        else:
            files = [f for f in os.listdir(daily_root) if f.endswith(".parquet")]
            tickers = [f.replace(".parquet", "") for f in files]
            
        logger.info(f"Generating priors for {len(tickers)} tickers from {args.start_date} to {args.end_date}")
        
        # 3. Processing Loop
        # Efficient pattern:
        # Iterate Dates -> Batch Tickers? 
        # Or Iterate Tickers -> All Dates?
        # Ticker-major is faster for data loading (open parquet once).
        # But we need to save partitioned by DATE.
        # So we collect results in memory buffer, then flush to date partitions.
        # Memory Check: 2000 tickers * 250 days * 8 floats ~ 32MB. Trivial.
        # We can process all tickers, hold results, then pivot and write.
        
        # BUT: Loading 2000 files is slow.
        # Batching:
        # Process N tickers at a time.
        
        results = []
        
        dates = pd.date_range(args.start_date, args.end_date, freq="B")
        req_dates = set([d.date() for d in dates])
        
        # Context window
        # We need T-252 to T.
        context_len = config.TEACHER_CONTEXT_DAYS
        
        batch_size = args.batch_size
        
        for t_idx, ticker in enumerate(tqdm(tickers)):
            # Load Data
            # raw ohlcv
            # We need log returns.
            # Load OHLCV, compute log returns.
            try:
                df = ohlcv_daily.load_ohlcv(ticker) # Returns pandas
                if df.empty: continue
                # Normalize timezone
                if df["date"].dt.tz is not None:
                    df["date"] = df["date"].dt.tz_localize(None)                
                df["date"] = df["date"].dt.normalize()
                # Check coverage
                # filter to range [start - context - buffer, end]
                # We need enough history for the first date.
                
                # Calculate log returns
                # ln(C_t / C_{t-1})
                df["close_adj"] = df["close_adj"].replace(0, np.nan).ffill()
                df["log_ret"] = np.log(df["close_adj"] / df["close_adj"].shift(1))
                df = df.dropna(subset=["log_ret"])
                
                # Make tensor
                # We need to map Dates -> Index
                # reindex to business days?
                # simplest: dict map date -> log_ret
                
                # Pre-compute contexts for requested dates
                ticker_results = []
                
                valid_rows = df[df["date"].isin(req_dates)]
                if valid_rows.empty: continue
                
                # We iterate relevant dates for this ticker
                # For each date D, we need window [D-context+1 : D] (inclusive of D? No, we predict from D close? 
                # "At close of D, predict D+1...". So context includes D.)
                
                # Indices
                # map date to row index
                date_to_idx = {d: i for i, d in enumerate(df["date"].dt.date)}
                values = torch.tensor(df["log_ret"].values, dtype=torch.float32, device=device)
                
                # Gather contexts
                batch_contexts = []
                batch_dates = []
                
                for d in req_dates:
                    if d not in date_to_idx: continue
                    idx = date_to_idx[d]
                    
                    if idx < context_len - 1: continue
                    
                    # Window: [idx - context_len + 1 : idx + 1]
                    # length = context_len
                    window = values[idx - context_len + 1 : idx + 1]
                    
                    # Normalize?
                    # Chronos expects raw values? Or normalized?
                    # Chronos usually handles scaling internally (Main Scaling).
                    # But we passed log returns.
                    # Standard practice: Pass raw log returns.
                    
                    batch_contexts.append(window)
                    batch_dates.append(d)
                    
                    if len(batch_contexts) >= batch_size:
                        # Flush batch (Vectorized inference for this ticker's dates)
                        # Actually Chronos batching usually across tickers.
                        # Here we batch across time for same ticker.
                        # Contexts might vary slightly (rolling window).
                        
                        c_tensor = torch.stack(batch_contexts) # [B, T]
                        
                        out = generate_priors_batch(wrapper, c_tensor, device)
                        
                        # Store
                        for i in range(len(batch_dates)):
                            bd = batch_dates[i]
                            res = {k: v[i] for k, v in out.items()}
                            res["date"] = bd
                            res["ticker"] = ticker
                            results.append(res)
                            
                        batch_contexts = []
                        batch_dates = []

                # Flush remainder
                if batch_contexts:
                    c_tensor = torch.stack(batch_contexts)
                    out = generate_priors_batch(wrapper, c_tensor, device)
                    for i in range(len(batch_dates)):
                        bd = batch_dates[i]
                        res = {k: v[i] for k, v in out.items()}
                        res["date"] = bd
                        res["ticker"] = ticker
                        results.append(res)

            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}")
                continue

        # 4. Save
        if not results:
            logger.warning("No priors generated!")
            return
            
        logger.info(f"Saving {len(results)} prior records...")
        out_df = pl.DataFrame(results)
        
        # Metadata
        meta = {
            "adapter": args.adapter_path,
            "horizon": config.TEACHER_FORECAST_HORIZON_DAYS,
            "context": config.TEACHER_CONTEXT_DAYS,
            "seed": args.seed,
            "tickers": len(tickers),
            "rows": len(out_df)
        }
        
        priors_store.write_priors_frame(out_df, args.out_dir, meta)
        
        run_recorder.finalize_run(run_id, "SUCCESS")
        
    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        run_recorder.finalize_run(run_id, "FAILED")
        sys.exit(1)

if __name__ == "__main__":
    main()
