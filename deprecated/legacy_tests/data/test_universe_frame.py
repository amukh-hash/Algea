
import pytest
import polars as pl
import pandas as pd
import numpy as np
from backend.app.data.universe_config import UniverseRules
from backend.app.data.universe_frame import UniverseBuilder

def test_universe_builder_rolling_metrics():
    """Test standard rolling metrics on synthetic data."""
    # 3 days of data for 2 tickers
    # Ticker A: consistent volume
    # Ticker B: zero volume
    
    dates = [
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-01-02"),
        pd.Timestamp("2023-01-03"),
    ]
    
    data = [
        {"date": dates[0], "ticker": "A", "close": 10.0, "volume": 1000.0}, # dvol 10k
        {"date": dates[1], "ticker": "A", "close": 10.0, "volume": 1000.0}, # dvol 10k
        {"date": dates[2], "ticker": "A", "close": 10.0, "volume": 1000.0}, # dvol 10k
        
        {"date": dates[0], "ticker": "B", "close": 5.0, "volume": 0.0},
        {"date": dates[1], "ticker": "B", "close": 5.0, "volume": 0.0},
        {"date": dates[2], "ticker": "B", "close": 5.0, "volume": 0.0},
    ]
    
    df = pl.DataFrame(data).with_columns(pl.col("date").cast(pl.Date))
    
    # Use small windows for testing
    rules = UniverseRules(
        adv20_window=2, # short window
        adv60_window=2, 
        min_age_observable=0 # disable age check
    )
    
    builder = UniverseBuilder(rules)
    res = builder.build(df)
    
    # Check A
    # adv20 (window=2) at day 3 (index 2) should be avg of day 2 and 3? 
    # Polars rolling_mean(2) at index i includes i and i-1.
    # Day 0: dvol=10k, adv20=null (if min_periods=window?) Polars rolling defaults?
    # Polars rolling_mean: window_size. 
    # rows:
    # A day 0: 10k -> adv20 null (count 1 < 2)
    # A day 1: 10k -> adv20 10k (count 2)
    # A day 2: 10k -> adv20 10k
    
    res_a = res.filter(pl.col("ticker") == "A").sort("date")
    assert res_a["adv20"][2] == 10000.0
    
    # Check B
    res_b = res.filter(pl.col("ticker") == "B")
    assert res_b["zero_volume_60"][2] >= 2 # 3 days of zero volume

def test_lookahead_bias():
    """Ensure metrics at T do not change if T+1 data changes."""
    dates = [
        pd.Timestamp("2023-01-01"),
        pd.Timestamp("2023-01-02"),
    ]
    
    # Base data up to T (Jan 1)
    # We provide T and T+1, but we check metric at T.
    data_1 = [
        {"date": dates[0], "ticker": "A", "close": 10.0, "volume": 100.0},
        {"date": dates[1], "ticker": "A", "close": 10.0, "volume": 100.0}, # T+1 is stable
    ]
    
    data_2 = [
        {"date": dates[0], "ticker": "A", "close": 10.0, "volume": 100.0},
        {"date": dates[1], "ticker": "A", "close": 20.0, "volume": 9999.0}, # T+1 spikes
    ]
    
    rules = UniverseRules(adv20_window=1, vol20_window=2)
    builder = UniverseBuilder(rules)
    
    res_1 = builder.build(pl.DataFrame(data_1).with_columns(pl.col("date").cast(pl.Date)))
    res_2 = builder.build(pl.DataFrame(data_2).with_columns(pl.col("date").cast(pl.Date)))
    
    # Metric at T (index 0) must be identical
    # Note: Polars rolling might need sorting. Builder does sorting.
    
    # Check metric at date[0]
    metric_1 = res_1.filter(pl.col("date") == dates[0].date())["adv20"][0]
    metric_2 = res_2.filter(pl.col("date") == dates[0].date())["adv20"][0]
    
    # If lookahead, metric_2 might be affected by T+1 spike? 
    # Rolling window (left aligned or right aligned?) 
    # Polars rolling default is "backward" (right aligned, i, i-1, ...).
    # So T depends on T, T-1. Not T+1.
    
    assert metric_1 == metric_2
    
    # Check metric at T+1
    metric_1_next = res_1.filter(pl.col("date") == dates[1].date())["adv20"][0]
    metric_2_next = res_2.filter(pl.col("date") == dates[1].date())["adv20"][0]
    
    assert metric_2_next > metric_1_next # T+1 should reflect T+1 data

def test_gates():
    """Test Observable vs Tradable logic."""
    # Single row check (since rolling logic verified above)
    # We construct a DF where rolling metrics are already "computed" 
    # (By faking the input data to produce specific averages? Or just trust the builder integration?)
    # Easier: Feed enough data to trigger conditions.
    
    # Tradable requires price >= 7.
    # Observable requires price >= 5.
    
    rules = UniverseRules(
        min_price_observable=5.0,
        min_price_tradable=7.0,
        min_age_observable=0,
        min_age_tradable=0,
        min_adv20_observable=0,
        min_adv60_observable=0,
        min_adv20_tradable=0,
        min_adv60_tradable=0,
        max_missing_60_observable=999,
        max_zero_volume_60_observable=999,
        max_missing_60_tradable=999,
        max_zero_volume_60_tradable=999,
        # Set all windows small to avoid nulls but large enough to pass max_share (1/4 = 0.25 < 0.30)
        adv20_window=4,
        adv60_window=4,
        vol20_window=4,
        missing_window=4,
        spike_window=4
    )
    
    # Need >1 day for vol20 to be finite? Or at least returns non-null.
    # Provide 10 days to clear chained rolling windows (adv20 then max_share)
    base_date = pd.Timestamp("2023-01-01")
    data = []
    for i in range(10):
        d = base_date + pd.Timedelta(days=i)
        data.extend([
            {"date": d, "ticker": "LOW", "close": 4.0, "volume": 1000.0},
            {"date": d, "ticker": "MID", "close": 6.0, "volume": 1000.0},
            {"date": d, "ticker": "HIGH", "close": 8.0, "volume": 1000.0},
        ])
    
    df = pl.DataFrame(data).with_columns(pl.col("date").cast(pl.Date))
    builder = UniverseBuilder(rules)
    res = builder.build(df)
    
    # Filter to last day (idx 9)
    last_day = base_date + pd.Timedelta(days=9)
    res_last = res.filter(pl.col("date") == last_day.date())
    
    # Verify masks
    low = res_last.filter(pl.col("ticker") == "LOW")
    assert not low["is_observable"][0]
    assert not low["is_tradable"][0]
    
    mid = res_last.filter(pl.col("ticker") == "MID")
    assert mid["is_observable"][0]
    assert not mid["is_tradable"][0]
    
    high = res_last.filter(pl.col("ticker") == "HIGH")
    assert high["is_observable"][0]
    assert high["is_tradable"][0]
