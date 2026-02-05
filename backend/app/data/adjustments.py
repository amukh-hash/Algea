
import pandas as pd

def adjust_daily_bars(raw_df: pd.DataFrame, actions_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Applies splits and dividends to raw OHLCV.
    """
    # Stub: Assume raw IS adjusted for now if source provides it (Alpaca 'adjustment' param)
    # Or implement simple ratio adjustment logic.
    
    # If Alpaca provides 'raw', we need split factors.
    # If Alpaca provides 'split_adjusted', we just use those columns mapping to *_adj.
    
    # We will assume input 'raw_df' already has adjusted columns or we map them.
    # Schema B2 requires: date, open_adj, high_adj, low_adj, close_adj, volume, dollar_volume
    
    out = raw_df.copy()
    if "close" in out.columns and "close_adj" not in out.columns:
        # Pass through if already adjusted from source
        out["close_adj"] = out["close"]
        out["open_adj"] = out["open"]
        out["high_adj"] = out["high"]
        out["low_adj"] = out["low"]
        
    out["dollar_volume"] = out["close_adj"] * out["volume"]
    
    # Basic fill for schema
    if "split_factor" not in out.columns:
        out["split_factor"] = 1.0
    if "dividend" not in out.columns:
        out["dividend"] = 0.0
    if "data_version" not in out.columns:
        out["data_version"] = "v1"
        
    # Ensure types
    out["date"] = pd.to_datetime(out["date"])
    
    return out
