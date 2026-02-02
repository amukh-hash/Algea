import polars as pl
import os
from typing import Optional, Tuple, Dict, Any

REQUIRED_SCHEMA = {
    "timestamp": pl.Datetime,
    "open": pl.Float64, # Or Float32 if we want to save space, user said float32 recommended for preproc outputs, but input might be 64.
    "high": pl.Float64,
    "low": pl.Float64,
    "close": pl.Float64,
    "volume": pl.Float64, # or Int64? Usually float after adjustments. Let's use Float64 for safety.
    "ad_line": pl.Float64,
    "bpi": pl.Float64
}

def load_ticker_data(ticker: str, data_dir: str) -> pl.DataFrame:
    fpath = os.path.join(data_dir, f"{ticker}_1m.parquet")
    if not os.path.exists(fpath):
        raise FileNotFoundError(f"Data for {ticker} not found at {fpath}")

    # Lazy load or eager? Eager is fine for 1 ticker.
    df = pl.read_parquet(fpath)

    # Ensure columns lower case
    df = df.rename({c: c.lower() for c in df.columns})

    # Ensure index is treated as column 'timestamp' if it's not
    if "timestamp" not in df.columns:
        # If it was saved from pandas with index, parquet might have it as __index_level_0__ or similar,
        # or just a named column if reset_index() was called.
        # Assuming typical pandas to_parquet without index=False, index is preserved.
        # Polars read_parquet usually reads index as a column if it has a name, or ignore it?
        # Let's inspect columns.
        # If standard schema isn't met, try to fix.
        pass

    return df

def load_breadth_context(context_path: str) -> pl.DataFrame:
    if not os.path.exists(context_path):
        raise FileNotFoundError(f"Breadth context not found at {context_path}")
    return pl.read_parquet(context_path)

def build_marketframe(ticker: str, data_dir: str, context_path: str) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """
    Builds the MarketFrame for a single ticker.
    Returns DataFrame and Metadata dict.
    """
    # 1. Load OHLCV
    ohlcv = load_ticker_data(ticker, data_dir)

    # Standardize OHLCV
    # Ensure 'timestamp' exists. If pandas index was used, it might be unnamed.
    # We assume 'timestamp' is available or we can infer it.
    # For now, let's assume the parquet has a timestamp column or we might need to handle it.
    # If the input parquet was saved with index='timestamp', polars might handle it specific way.

    # Let's try to find the datetime column
    time_col = None
    for col, dtype in ohlcv.schema.items():
        if dtype in (pl.Datetime, pl.Date):
            time_col = col
            break

    if time_col:
        ohlcv = ohlcv.rename({time_col: "timestamp"})
    else:
        # Fallback: look for strings that look like dates? No, assume correct schema for now.
        if "index" in ohlcv.columns:
             ohlcv = ohlcv.rename({"index": "timestamp"})
        # raise ValueError("Could not identify timestamp column in OHLCV")

    # Select only required columns from OHLCV
    req_ohlcv = ["timestamp", "open", "high", "low", "close", "volume"]
    # Cast to Float64
    ohlcv = ohlcv.select([
        pl.col("timestamp"),
        pl.col("open").cast(pl.Float64),
        pl.col("high").cast(pl.Float64),
        pl.col("low").cast(pl.Float64),
        pl.col("close").cast(pl.Float64),
        pl.col("volume").cast(pl.Float64)
    ])

    # 2. Load Breadth
    breadth = load_breadth_context(context_path)

    # Ensure timestamp column in breadth
    # Similar check
    if "timestamp" not in breadth.columns:
        # If pandas index, usually it becomes a column or index column
        # Let's assume it's properly named or first column
        # If saved via pandas to_parquet, index is usually preserved.
        pass # assume ok for now

    # 3. Join
    # Left join on OHLCV timestamp.
    # We want to Keep OHLCV timestamps, and attach Breadth.
    # If Breadth is missing (e.g. gaps in global context?), we ffill?
    # Plan says: "Breadth data: forward-fill within session only (or across sessions if you want — but pick one)"
    # Plan also says: "MarketFrame join policy: strict alignment/ffill"

    # If we do left join, we get nulls where breadth is missing.
    # Then we ffill breadth.

    mf = ohlcv.join(breadth, on="timestamp", how="left")

    # Sort by timestamp
    mf = mf.sort("timestamp")

    # FFill breadth columns (ad_line, bpi)
    mf = mf.with_columns([
        pl.col("ad_line").forward_fill(),
        pl.col("bpi").forward_fill()
    ])

    # 4. Missing Data Policy
    # "Drop timestamps where OHLCV is missing" -> We started with OHLCV keys, so we only have timestamps where OHLCV exists.
    # But if OHLCV has nulls?
    # We should drop rows where OHLCV is null.

    initial_rows = mf.height
    mf = mf.drop_nulls(subset=["open", "high", "low", "close", "volume"])
    final_rows = mf.height

    missing_count = initial_rows - final_rows

    # 5. Schema Check
    # Verify columns exist
    for col in REQUIRED_SCHEMA:
        if col not in mf.columns:
            raise ValueError(f"Missing column {col} in MarketFrame")

    metadata = {
        "ticker": ticker,
        "rows": final_rows,
        "dropped_missing_ohlcv": missing_count,
        "columns": mf.columns
    }

    return mf, metadata
