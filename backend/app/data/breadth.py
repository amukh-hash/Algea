import pandas as pd
import numpy as np

def calculate_ad_line(closes: pd.DataFrame) -> pd.Series:
    """
    Calculates Advance/Decline Line (AD_Line).
    (Count Up - Count Down) / Total
    Input: DataFrame where columns are Tickers, rows are Close prices.
    """
    # Periodic Returns (1 min)
    returns = closes.diff()
    
    up_counts = (returns > 0).sum(axis=1)
    down_counts = (returns < 0).sum(axis=1)
    # total valid for that minute (not NaN)
    valid_counts = returns.count(axis=1)
    
    # Avoid division by zero
    valid_counts = valid_counts.replace(0, np.nan)
    
    ad_line = (up_counts - down_counts) / (valid_counts)
    return ad_line.fillna(0.0)

def calculate_bpi(dfs: dict, master_index: pd.DatetimeIndex) -> pd.Series:
    """
    Calculates Buying Pressure Index (BPI) using CLV.
    Mean CLV of all stocks.
    """
    clv_sum = pd.Series(0.0, index=master_index)
    count = pd.Series(0, index=master_index)
    
    for t, df in dfs.items():
        # Align
        # Assumes df is already somewhat clean, but reindexing ensures alignment
        aligned = df.reindex(master_index).ffill(limit=5)
        
        C = aligned['close']
        H = aligned['high']
        L = aligned['low']
        
        # CLV = ((C - L) - (H - C)) / (H - L)
        # Avoid div by zero (H=L)
        rnge = (H - L).replace(0, np.nan)
        clv = ((C - L) - (H - C)) / rnge
        
        clv = clv.fillna(0) # No range = Neutral pressure
        
        clv_sum = clv_sum.add(clv, fill_value=0)
        
        # Increment count where we have valid data (original df had data or filled)
        # We check if 'close' was not NaN after reindex+ffill
        not_nan = aligned['close'].notna().astype(int)
        count = count.add(not_nan, fill_value=0)
    
    # Avoid div by zero
    count = count.replace(0, np.nan)
    bpi = clv_sum / count
    return bpi.fillna(0.0)
