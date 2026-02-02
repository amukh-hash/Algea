import os
import sys
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from chronos import ChronosPipeline

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Constants
OUTPUT_PATH = "backend/data/judge_training_data.csv"
MODEL_ID = "amazon/chronos-bolt-small" # Judge trains on Student errors
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

from app.api.databento_client import DatabentoClient
from app.targets.triple_barrier import get_daily_vol, apply_triple_barrier

def generate_data():
    print(f"--- Generating Judge Data using {MODEL_ID} on {DEVICE} ---")
    
    # 1. Fetch Data (Use CSV or Databento Mock)
    # Using Databento Mock for consistency with other scripts
    client = DatabentoClient(mock_mode=True)
    df = client.get_historical_range("BTC-USD", "2023-01-01", "2023-03-01", schema='mbp-10')
    
    if 'price' in df.columns:
        prices_series = df['price']
    else:
        prices_series = (df['bid_px_00'] + df['ask_px_00']) / 2.0
    
    prices = prices_series.values
    print(f"Loaded {len(prices)} ticks.")

    # 2. Pipeline
    pipeline = ChronosPipeline.from_pretrained(
        MODEL_ID,
        device_map=DEVICE,
        torch_dtype=torch.float32
    )

    # 3. Triple Barrier Labels (Ground Truth)
    print("Calculating Triple Barriers...")
    vol = get_daily_vol(prices_series, span=100)
    
    # Horizon 32 for Bolt vs T5's 64?
    # Live script uses predict(1). But decision horizon is usually longer.
    # Let's align with Distillation: Context 64, Pred 32.
    pred_len = 32 
    context_len = 64
    
    events = apply_triple_barrier(
        prices_series, 
        vol, 
        vertical_barrier_steps=pred_len, 
        barrier_width_multiplier=2.0 
    )
    # events: label (0,1,2), ret
    
    # 4. Generate Predictions & Features
    # Stride to avoid high overlap
    stride = 32
    valid_indices = range(context_len, len(prices) - pred_len, stride)
    
    BATCH_SIZE = 8
    
    # Check resume
    if os.path.exists(OUTPUT_PATH):
        print("Overwriting existing data for consistency...")
    
    # Header
    # Features: volatility, uncertainty_spread
    with open(OUTPUT_PATH, 'w') as f:
        f.write("true_label,true_ret,params_q50,uncertainty_spread,volatility,is_correct\n")

    print(f"Inference on {len(valid_indices)} samples...")
    
    processed = 0
    buffer_rows = []
    
    idx_list = list(valid_indices)
    
    for i in tqdm(range(0, len(idx_list), BATCH_SIZE)):
        batch_idxs = idx_list[i : i+BATCH_SIZE]
        
        # Contexts
        contexts = [torch.tensor(prices[idx-context_len : idx], dtype=torch.float32) for idx in batch_idxs]
        
        try:
            # Predict Quantiles [0.1, 0.5, 0.9]
            # Verify Bolt pipeline behavior.
            # If `num_samples` provided, it samples.
            # We want direct quantiles if possible, but sampling is robust.
            forecasts = pipeline.predict(contexts, prediction_length=pred_len, num_samples=20)
            # (B, Samples, Horizon)
        except Exception as e:
            print(f"Inference error: {e}")
            continue
            
        # Process
        for b, idx in enumerate(batch_idxs):
            # Target
            # events aligns with prices index.
            # Entry at `idx-1`.
            evt_idx = idx - 1
            if evt_idx not in events.index: continue
            
            true_label = int(events.loc[evt_idx, 'label']) # 0, 1, -1? TBM returns -1 for Sell
            true_ret = events.loc[evt_idx, 'ret']
            
            # TBM returns: 1 (Buy), -1 (Sell), 0 (Neutral)
            # Map -1 to 2 for classification? Or keep -1.
            # XGBoost handles negative labels? Usually 0,1,2.
            # Let's map -1 -> 2.
            if true_label == -1: true_label = 2
            
            # Forecast Analysis
            f_samples = forecasts[b].numpy() # (20, 32)
            
            # Calculate Quantiles over samples
            q10 = np.quantile(f_samples, 0.1, axis=0).mean()
            q50 = np.quantile(f_samples, 0.5, axis=0).mean() # Mean over horizon
            q90 = np.quantile(f_samples, 0.9, axis=0).mean()
            
            # Uncertainty Spread (IQR)
            # (q90-q10) / |q50|
            # Avoid div zero
            denom = abs(q50) if abs(q50) > 1e-6 else 1.0
            spread = (q90 - q10) / denom
            
            # Volatility (Feature)
            current_vol = vol.loc[evt_idx] if not np.isnan(vol.loc[evt_idx]) else 0.0
            
            # Derived Prediction
            # If q50 > price_at_entry?
            price_entry = prices[idx-1]
            pred_ret = (q50 - price_entry) / price_entry
            
            pred_label = 0
            thresh = 0.001
            if pred_ret > thresh: pred_label = 1
            elif pred_ret < -thresh: pred_label = 2
            
            # Is Correct?
            is_correct = 1 if pred_label == true_label and pred_label != 0 else 0
            # If Neutral match, is it "Correct"? 
            # Usually Judge filters for active signals.
            
            row = f"{true_label},{true_ret:.6f},{pred_ret:.6f},{spread:.6f},{current_vol:.6f},{is_correct}\n"
            buffer_rows.append(row)
            
    with open(OUTPUT_PATH, 'a') as f:
        for r in buffer_rows:
            f.write(r)
            
    print(f"Saved {len(buffer_rows)} rows to {OUTPUT_PATH}")

if __name__ == "__main__":
    generate_data()
