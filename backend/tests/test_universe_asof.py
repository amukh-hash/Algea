
import pytest
import polars as pl
import pandas as pd
import numpy as np
from datetime import date, timedelta
from backend.app.data.universe_frame import UniverseBuilder

def create_synthetic_ohlcv(symbol="TEST", start_date=date(2020, 1, 1), end_date=date(2020, 6, 1)):
    dates = pd.date_range(start_date, end_date, freq="B")
    n = len(dates)
    
    # Random walk
    np.random.seed(42)
    returns = np.random.normal(0, 0.02, n)
    price = 100 * np.exp(np.cumsum(returns))
    volume = np.random.randint(1000, 10000, n) * 100.0
    
    df = pl.DataFrame({
        "date": [d.date() for d in dates],
        "ticker": [symbol] * n,
        "close": price,
        "volume": volume,
        # close_prev isn't strictly needed if builder computes returns
    })
    return df

def test_universe_asof_invariance():
    """
    Test that universe metrics at date T depend ONLY on data <= T.
    """
    full_df = create_synthetic_ohlcv(end_date=date(2021, 1, 1))
    
    # Add a "future shock" to verify we catch leakage if it exists
    # If we modify data at T+1, T should NOT change.
    
    builder = UniverseBuilder()
    
    # 1. Full Build
    full_frame = builder.build(full_df)
    
    # Pick a test date T
    test_date_idx = len(full_df) // 2
    test_date = full_df["date"][test_date_idx]
    
    print(f"Testing As-Of invariance at {test_date}")
    
    # Get stats from full build at T
    row_full = full_frame.filter(pl.col("date") == test_date).to_dicts()[0]
    
    # 2. As-Of Build (Data truncated strictly at T)
    asof_df = full_df.filter(pl.col("date") <= test_date)
    asof_frame = builder.build(asof_df)
    row_asof = asof_frame.filter(pl.col("date") == test_date).to_dicts()[0]
    
    # 3. Compare Keys
    # We care about rolling metrics: adv20, adv60, vol20, etc.
    keys_to_check = ["adv20", "adv60", "vol20", "max_share20", "is_tradable", "is_observable"]
    
    errors = []
    for k in keys_to_check:
        v_full = row_full.get(k)
        v_asof = row_asof.get(k)
        
        # Handle float comparison
        if isinstance(v_full, float) and (np.isnan(v_full) or np.isinf(v_full)):
             # As-of should match
             if not (np.isnan(v_asof) or np.isinf(v_asof)):
                 errors.append(f"{k} mismatch: Full={v_full}, AsOf={v_asof}")
        elif isinstance(v_full, float):
             if abs(v_full - v_asof) > 1e-6:
                 errors.append(f"{k} mismatch: Full={v_full}, AsOf={v_asof}")
        else:
             if v_full != v_asof:
                 errors.append(f"{k} mismatch: Full={v_full}, AsOf={v_asof}")
                 
    assert not errors, "\n".join(errors)

def test_future_leakage_sensitivity():
    """
    Modify data at T+1 and ensure T does NOT change.
    """
    builder = UniverseBuilder()
    df = create_synthetic_ohlcv(end_date=date(2021, 1, 1))
    test_date_idx = 100
    test_date = df["date"][test_date_idx]
    
    # Baseline
    res_base = builder.build(df).filter(pl.col("date") == test_date).to_dicts()[0]
    
    # Pivot: Modify T+1 volume/price drastically
    # This should affect T+1 rolling stats, but NOT T.
    
    # Polars is immutable, create new df
    df_mod = df.with_columns(
        pl.when(pl.col("date") == df["date"][test_date_idx + 1])
        .then(pl.col("volume") * 10000) # Massive splice
        .otherwise(pl.col("volume"))
        .alias("volume")
    )
    
    res_mod = builder.build(df_mod).filter(pl.col("date") == test_date).to_dicts()[0]
    
    assert res_base["adv20"] == res_mod["adv20"], "ADV20 at T changed when T+1 volume modified! LEAKAGE DETECTED."
    assert res_base["vol20"] == res_mod["vol20"], "VOL20 at T changed when T+1 modified!"

if __name__ == "__main__":
    test_universe_asof_invariance()
    test_future_leakage_sensitivity()
    print("ALL TESTS PASSED")
