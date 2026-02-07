import pandas as pd
import numpy as np
from backend.app.ops import run_recorder

class FeatureEngineer:
    """
    Transforms raw OHLCV into:
    1. Feature Vectors (for Rank-Transformer)
    2. Time Sequences (for Chronos-2 Inference)
    """
    
    def process_features(self, 
                         universe_df: pd.DataFrame, 
                         market_df: pd.DataFrame,
                         mode: str = 'inference',
                         run_id: str | None = None) -> pd.DataFrame:
        """
        Generates the standard scalar features for the Ranking Model.
        """
        # Sort is essential for rolling/shift ops
        df = universe_df.sort_values(['ticker', 'date']).copy()
        
        # --- FIX 1: Grouped Calculations ---
        # Previous bug: df['close'].shift(1) shifted across tickers.
        # Fix: operate per group.
        
        g = df.groupby('ticker')
        
        # Log Returns: ln(P_t / P_t-1)
        # We use .pct_change() then log1p ensures robustness
        df['log_ret'] = np.log1p(g['close'].pct_change())
        
        # 1. Core Returns
        df['log_return_1d'] = df['log_ret']
        df['log_return_5d'] = g['log_ret'].rolling(5).sum().reset_index(0, drop=True)
        df['log_return_20d'] = g['log_ret'].rolling(20).sum().reset_index(0, drop=True)
        
        # 2. Volatility (20d raw std dev)
        df['volatility_20d'] = g['log_ret'].rolling(20).std().reset_index(0, drop=True)
        
        # 3. Relative Volume
        # Avoid lookahead: Compare today's volume to the MEDIAN of the LAST 20 days (shift 1)
        # Using a closed window 'left' or shift(1) is safer.
        prior_vol_median = g['volume'].shift(1).rolling(20).median().reset_index(0, drop=True)
        df['relative_volume_20d'] = df['volume'] / (prior_vol_median + 1.0) # +1 avoids div/0
        
        # 4. Market Context (Merge)
        df = df.merge(market_df, on='date', how='left')
        
        # 5. Targets (Train Only)
        if mode == 'train':
            # Target: Return from Close_t to Close_t+10
            # Shift(-10) brings the future value back to current row
            fwd_close = g['close'].shift(-10)
            df['target_10d_fwd'] = np.log(fwd_close / df['close'])
            df.dropna(subset=['target_10d_fwd'], inplace=True)

        # 6. Burn-in Cleanup
        # Remove rows where features are NaN (first 20 days)
        df.dropna(subset=['log_return_20d', 'volatility_20d'], inplace=True)
        
        if run_id:
            run_recorder.emit_event(
                run_id,
                stage="preproc",
                step="process_features",
                level="INFO",
                message="Feature engineering completed",
                payload={"rows": len(df), "mode": mode},
            )
        return df

    def get_chronos_sequences(self, 
                              df: pd.DataFrame, 
                              lookback: int = 512) -> pd.DataFrame:
        """
        Extracts the last 'lookback' days for EACH ticker for Chronos Inference.
        Returns a DataFrame where one row = one ticker, containing list columns.
        """
        # We need the most recent 'lookback' rows for every ticker
        # Efficiency: slicing tail(lookback) per group
        
        subset = df.sort_values(['ticker', 'date']).groupby('ticker').tail(lookback)
        
        # Validate lengths: If a ticker has < lookback, Chronos might fail or pad.
        # We will pad with 0s or handle in the codec, but here we just aggregate.
        
        # Aggregate into lists
        seqs = subset.groupby('ticker').agg({
            'log_ret': list,
            # We assume we might add covariates here later
        }).reset_index()
        
        # Rename for clarity
        seqs.rename(columns={'log_ret': 'sequence_returns'}, inplace=True)
        return seqs
