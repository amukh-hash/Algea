import pytest
import polars as pl
import numpy as np
import pandas as pd
from backend.app.data import windows, calendar

def test_swing_window_dataset_targets():
    # Setup: 5 days of data (Mon-Fri), 10:00 to 16:00, hourly bars for simplicity (or minutely but sparse)
    # Mon Oct 16 2023 to Fri Oct 20 2023.
    # Sessions: Mon 16, Tue 17, Wed 18, Thu 19, Fri 20.
    
    # Create timestamps
    # We need explicit Close times: 16:00 ET -> 20:00 UTC.
    dates = []
    prices = []
    
    # Mon
    dates.append("2023-10-16 14:00:00") # 10 AM ET (Open) - Price 100
    prices.append(100.0)
    dates.append("2023-10-16 20:00:00") # 4 PM ET (Close) - Price 101
    prices.append(101.0)
    
    # Tue
    dates.append("2023-10-17 20:00:00") # Close - Price 102
    prices.append(102.0)
    
    # Wed
    dates.append("2023-10-18 20:00:00") # Close - Price 103
    prices.append(103.0)
    
    # Thu
    dates.append("2023-10-19 20:00:00") # Close - Price 104
    prices.append(104.0)

    df = pl.DataFrame({
        "timestamp": pd.to_datetime(dates).tz_localize("UTC"),
        "close": prices,
        "feature1": prices # dummy feature
    })
    
    # Dataset config
    lookback = 1
    input_cols = ["feature1"]
    
    ds = windows.SwingWindowDataset(df, input_cols, lookback, stride=1, horizons=["1D", "3D"])
    
    # We expect valid sample for Mon 10:00 (index 0).
    # 1D Target: Mon Close (index 1, Price 101).
    # 3D Target: Wed Close (index 3, Price 103).
    
    # Index 0 is Mon 10:00.
    # Is it valid?
    # valid_indices should contain index 0?
    # lookback=1 -> start_idx = 0.
    
    assert len(ds) >= 1
    
    # Find sample corresponding to Mon 10:00
    # The valid_indices logic:
    # i=0. Target 1D -> Mon Close (20:00). exists at idx 1.
    # Target 3D -> Wed Close. exists at idx 3.
    # So index 0 is valid.
    
    # Let's get the item corresponding to index 0 (which is likely the first valid item)
    # Check ds.valid_indices
    assert 0 in ds.valid_indices
    
    # Get item
    # Since ds is a map-style dataset, we index by 0..len-1.
    # We need to find which "logical" index corresponds to real index 0.
    logical_idx = ds.valid_indices.index(0)
    
    x, y = ds[logical_idx]
    
    # Check X
    # Lookback 1. x should be feature at index 0.
    assert x.shape == (1, 1)
    assert x[0, 0] == 100.0
    
    # Check Y
    # 1D: log(101/100)
    expected_1d = np.log(101.0/100.0)
    assert np.allclose(y["1D"], expected_1d)
    
    # 3D: log(103/100)
    expected_3d = np.log(103.0/100.0)
    assert np.allclose(y["3D"], expected_3d)

def test_swing_window_dataset_lookback_alignment():
    # Test that window x is correctly aligned [t-L+1 : t+1]
    # Timestamps: 0, 1, 2, 3, 4
    # Prices: 10, 11, 12, 13, 14
    dates = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    df = pl.DataFrame({
        "timestamp": dates,
        "close": [10, 11, 12, 13, 14],
        "f": [0, 1, 2, 3, 4]
    })
    
    # Mock compute_targets to assume all valid to test X alignment only
    # We can override _compute_targets or just ensure targets exist.
    # Since dates are arbitrary, get_next_session_close might fail or return far future.
    # Let's trust logic but use a mock calendar? 
    # Or just subclass and override _compute_targets to mark all valid.
    
    class MockDataset(windows.SwingWindowDataset):
        def _compute_targets(self):
            # Mark all >= lookback-1 as valid
            self.targets = {"1D": np.zeros(len(self.timestamps))} # dummy
            self.valid_indices = list(range(self.lookback-1, len(self.timestamps)))
            
    ds = MockDataset(df, ["f"], lookback=3, horizons=["1D"])
    
    # First valid index is 2 (need 0,1,2).
    # x should be [f[0], f[1], f[2]] -> [0, 1, 2]
    
    x, _ = ds[0] # logical index 0 -> real index 2
    
    assert x.shape == (3, 1)
    assert x[0, 0] == 0
    assert x[2, 0] == 2
