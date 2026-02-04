import pytest
import polars as pl
import pandas as pd
import numpy as np
import os
from backend.app.data import marketframe

def test_marketframe_alignment(tmp_path):
    # Setup dummy data
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    context_path = tmp_path / "breadth.parquet"
    
    # Create OHLCV for Ticker 'TEST'
    # Timestamps: 10:00, 10:01, 10:03 (gap), 10:04
    dates = pd.to_datetime(["2023-01-01 10:00", "2023-01-01 10:01", "2023-01-01 10:03", "2023-01-01 10:04"])
    ohlcv_df = pd.DataFrame({
        "timestamp": dates,
        "open": [100.0, 101.0, 103.0, 104.0],
        "high": [100.5, 101.5, 103.5, 104.5],
        "low": [99.5, 100.5, 102.5, 103.5],
        "close": [100.2, 101.2, 103.2, 104.2],
        "volume": [1000, 1100, 1300, 1400]
    })
    # Save as parquet
    # We save with index=False usually, relying on column
    ohlcv_df.to_parquet(data_dir / "TEST_1m.parquet", index=False)
    
    # Create Breadth Context
    # Timestamps: 10:00, 10:01, 10:02, 10:03, 10:04 (Complete)
    # 10:02 exists in breadth but not in OHLCV -> Should NOT appear in output (Left Join)
    dates_ctx = pd.to_datetime(["2023-01-01 10:00", "2023-01-01 10:01", "2023-01-01 10:02", "2023-01-01 10:03", "2023-01-01 10:04"])
    breadth_df = pd.DataFrame({
        "timestamp": dates_ctx,
        "ad_line": [0.5, 0.6, 0.7, 0.8, 0.9],
        "bpi": [50.0, 51.0, 52.0, 53.0, 54.0]
    })
    breadth_df.to_parquet(context_path, index=False)
    
    # Run Build
    mf, meta = marketframe.build_marketframe("TEST", str(data_dir), str(context_path))
    
    # Verify Metadata
    assert meta["ticker"] == "TEST"
    assert meta["rows"] == 4 # Should match OHLCV count
    
    # Verify Join
    # Row 10:00 -> ad=0.5
    row0 = mf.filter(pl.col("timestamp") == pd.Timestamp("2023-01-01 10:00")).to_dict(as_series=False)
    assert row0["ad_line"][0] == 0.5
    
    # Row 10:03 -> ad=0.8
    row2 = mf.filter(pl.col("timestamp") == pd.Timestamp("2023-01-01 10:03")).to_dict(as_series=False)
    assert row2["ad_line"][0] == 0.8
    
    # Verify no 10:02
    assert mf.filter(pl.col("timestamp") == pd.Timestamp("2023-01-01 10:02")).height == 0

def test_marketframe_missing_ohlcv_handling(tmp_path):
    # Test that we drop rows if OHLCV has nulls (e.g. malformed data)
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    context_path = tmp_path / "breadth.parquet"
    
    dates = pd.to_datetime(["2023-01-01 10:00", "2023-01-01 10:01"])
    ohlcv_df = pd.DataFrame({
        "timestamp": dates,
        "open": [100.0, None], # Missing open
        "high": [100.5, 101.5],
        "low": [99.5, 100.5],
        "close": [100.2, 101.2],
        "volume": [1000, 1100]
    })
    ohlcv_df.to_parquet(data_dir / "BAD_1m.parquet", index=False)
    
    # Breadth
    breadth_df = pd.DataFrame({
        "timestamp": dates,
        "ad_line": [0.5, 0.6],
        "bpi": [50.0, 51.0]
    })
    breadth_df.to_parquet(context_path, index=False)
    
    mf, meta = marketframe.build_marketframe("BAD", str(data_dir), str(context_path))
    
    # Should have dropped the row with None
    assert meta["rows"] == 1
    assert meta["dropped_missing_ohlcv"] == 1
    assert mf.height == 1
