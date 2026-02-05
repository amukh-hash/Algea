import pandas as pd
import numpy as np

class FeatureContract:
    """
    The Single Source of Truth for Data Shapes.
    Enforces 'Strict Mode' for Ranking to prevent trading on partial data.
    """
    
    # --- Feature Groups ---
    
    CORE_FEATURES = [
        'symbol', 'date',
        'ret_1d', 'ret_5d', 'ret_20d',
        'volatility_20d', 'relative_volume_20d'
    ]
    
    MARKET_FEATURES = [
        'spy_ret_1d', 'vix_level', 'vix_ret_1d', 'market_breadth_ad'
    ]
    
    # The 'Teacher' Priors (Chronos-2 Outputs)
    # Must match EXACTLY what the nightly inference produces
    PRIOR_FEATURES = [
        'prior_drift_20d',
        'prior_vol_20d',
        'prior_downside_q10', # The 10th percentile outcome (Tail Risk)
        'prior_trend_conf'    # Probability of positive trend
    ]
    
    TARGETS = ['target_10d_fwd']

    @classmethod
    def validate(cls, df: pd.DataFrame, mode: str = 'inference') -> bool:
        """
        Args:
            mode: 'train' | 'ranking' (inference with priors) | 'pre-priors' (intermediate)
        """
        required = cls.CORE_FEATURES + cls.MARKET_FEATURES
        
        if mode == 'ranking':
            required += cls.PRIOR_FEATURES
        elif mode == 'train':
            required += cls.PRIOR_FEATURES + cls.TARGETS
            
        # 1. Missing Columns
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"SCHEMA ERROR: Missing columns: {missing}")
            
        # 2. Type Checks
        # Dates must be dates, not strings
        # Avoid mutating the input dataframe unless explicitly allowed. 
        # For validation, we check. If coercion is needed, the caller should handle it or we assume it fails.
        if not np.issubdtype(df['date'].dtype, np.datetime64):
             # Check if they look like dates without mutating
             try:
                 pd.to_datetime(df['date'], errors='raise')
             except:
                 raise TypeError("SCHEMA ERROR: 'date' column must be datetime64 and could not be coerced.")
             
             # If we want to strictly enforce types without implicit coercion in the object
             # raise TypeError("SCHEMA ERROR: 'date' column must be datetime64 (passed as object/string)")
             
             # User Request: "Avoid mutating input".
             # So we do NOT write back to df['date'].
             pass

        # 3. NaN Policy (The 'Zero Tolerance' Rule)
        # In ranking mode, any NaN in a feature = unknown behavior = DANGEROUS.
        if mode in ['ranking', 'train']:
            # Check only feature columns (ignore metadata if any exists)
            check_cols = [c for c in required if c != 'target_10d_fwd']
            if df[check_cols].isnull().any().any():
                bad_rows = df[df[check_cols].isnull().any(axis=1)]
                bad_tickers = bad_rows['ticker'].unique()
                raise ValueError(f"DATA INTEGRITY ERROR: NaNs detected for {len(bad_rows)} rows. "
                                 f"Affected tickers: {bad_tickers[:5]}...")

        return True
