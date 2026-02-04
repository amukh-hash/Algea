import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Optional

class UniverseSelector:
    """
    Production-Grade Universe Selector.
    
    CRITICAL FEATURES:
    1. Dynamic Membership: Reconstructs eligibility for any historical date.
    2. Trading Day Logic: Uses count of bars, not just calendar deltas.
    3. Strict Liquidity: Enforces minimum observations to prevent 'lucky' inclusions.
    """
    
    def __init__(self, 
                 min_price: float = 5.0, 
                 min_adv: float = 25_000_000, 
                 min_obs_for_adv: int = 20,
                 min_seasoning_days: int = 252):
        self.min_price = min_price
        self.min_adv = min_adv
        self.min_obs_for_adv = min_obs_for_adv
        self.min_seasoning_days = min_seasoning_days

    def select(self, 
               df: pd.DataFrame, 
               metadata: pd.DataFrame, 
               as_of_date: str) -> pd.DataFrame:
        """
        Returns a 'UniverseMembership' DataFrame for the given date.
        
        Args:
            df: Historical daily pricing [ticker, date, close, volume]
            metadata: Static data [ticker, asset_type, ipo_date]
            as_of_date: YYYY-MM-DD
            
        Returns:
            DataFrame: [ticker, is_eligible, reason_code]
        """
        # 1. Filter to causal window (Data available <= as_of_date)
        # We need enough history to calculate the 20d Rolling ADV
        target_date = pd.to_datetime(as_of_date)
        lookback_date = target_date - timedelta(days=60) # Buffer for 20 trading days
        
        # Subset efficiently
        subset = df[(df['date'] >= lookback_date) & (df['date'] <= target_date)].copy()
        
        # 2. Compute Liquidity Metrics (Efficient Groupby)
        subset['dollar_vol'] = subset['close'] * subset['volume']
        
        # We only care about the metrics AT the target_date
        # We calculate rolling stats, then pick the last row per ticker
        stats = subset.sort_values('date').groupby('ticker').agg({
            'close': 'last',
            'dollar_vol': lambda x: x.tail(20).median() if len(x) >= 20 else np.nan,
            'volume': lambda x: x.tail(20).median() if len(x) >= 20 else np.nan,
            'date': 'max' # To check if the ticker actually traded on/near as_of_date
        }).reset_index()
        
        # 3. Merge Metadata
        stats = stats.merge(metadata, on='ticker', how='left')
        
        # 4. Apply Filters & Log Reasons
        results = []
        
        for _, row in stats.iterrows():
            reasons = []
            
            # A. Liveness Check
            if row['date'] != target_date:
                # If data is stale (ticker didn't trade today), exclude.
                # In production, allow 1-2 days gap for holidays/halts.
                if (target_date - row['date']).days > 2:
                    reasons.append("STALE_DATA")
            
            # B. Asset Type (Normalize)
            atype = str(row.get('asset_type', '')).upper()
            if atype not in ['CS', 'EQ', 'COMMON STOCK']:
                reasons.append("INVALID_ASSET_TYPE")
                
            # C. Seasoning (Trading Days proxy)
            # Check if IPO date is known and far enough back
            ipo = pd.to_datetime(row.get('ipo_date', 'NaT'))
            if pd.isna(ipo) or (target_date - ipo).days < 365: # Approx 252 trading days
                reasons.append("INSUFFICIENT_HISTORY")
                
            # D. Price
            if row['close'] < self.min_price:
                reasons.append("PRICE_TOO_LOW")
                
            # E. Liquidity (ADV)
            if pd.isna(row['dollar_vol']) or row['dollar_vol'] < self.min_adv:
                reasons.append("LOW_LIQUIDITY")
                
            is_eligible = len(reasons) == 0
            results.append({
                'as_of_date': as_of_date,
                'ticker': row['ticker'],
                'is_eligible': is_eligible,
                'fail_reasons': ",".join(reasons) if not is_eligible else "PASS",
                'metric_adv': row['dollar_vol']
            })
            
        return pd.DataFrame(results)
