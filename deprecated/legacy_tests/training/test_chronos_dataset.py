"""
Test ChronosDataset Readiness Checklist (3.1 - 3.4)
"""
import pytest
import polars as pl
import numpy as np
import torch
from pathlib import Path
from backend.app.training.chronos_dataset import ChronosDataset

@pytest.fixture
def mock_obs_lookup(monkeypatch):
    """Mock loading UniverseFrame observable mask"""
    # Ticker A: Observable on days 10..20
    # Ticker B: Never observable
    def mock_load(path):
        return {
            "A": {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
            "B": set()
        }
    monkeypatch.setattr("backend.app.training.chronos_dataset.ChronosDataset._load_universe_mask", lambda self, p: mock_load(p))

def test_chronos_dataset_readiness(tmp_path, mock_obs_lookup):
    """
    Verify Checklist 3.1-3.4
    """
    # Create synthetic parquet files
    # Ticker A: Valid data
    df_a = pl.DataFrame({
        "date": list(range(30)), # Integer dates for simplicity in mock
        "close": [100.0 * (1.01**i) for i in range(30)]
    })
    
    # Ticker B: NaNs
    df_b = pl.DataFrame({
        "date": list(range(30)),
        "close": [100.0 if i != 15 else np.nan for i in range(30)]
    })
    
    file_a = tmp_path / "A.parquet"
    file_b = tmp_path / "B.parquet"
    df_a.write_parquet(file_a)
    df_b.write_parquet(file_b)
    
    # Init Dataset
    # Context=5, Pred=2, Stride=2
    ds = ChronosDataset(
        files=[file_a, file_b],
        context_len=5,
        prediction_len=2,
        stride=2,
        target_col="close"
    )
    
    # CHECK 3.1: Observable Mask
    # Ticker A is observable on 10..20.
    # Anchor dates must be in 10..20.
    # Window: Start S. Anchor = S + Context - 1 = S + 4.
    # S+4 must be in {10..20}.
    # So valid Anchors: 10, 11, ... 20.
    # Corresponding Starts: 6, 7, ... 16.
    # Stride=2. Valid Starts: 6, 8, 10, 12, 14, 16.
    
    # Ticker B is never observable -> Should have NO samples.
    
    # Let's inspect index
    # Format: (file_idx, start_row)
    # file_idx 0 is A, 1 is B
    
    indices_a = [x[1] for x in ds.index if x[0] == 0]
    indices_b = [x[1] for x in ds.index if x[0] == 1]
    
    print(f"Indices A: {indices_a}")
    print(f"Indices B: {indices_b}")
    
    assert len(indices_b) == 0, "Ticker B (Not Observable) should be excluded."
    
    # Verify A indices are correct subset
    expected_starts = {6, 8, 10, 12, 14, 16}
    assert set(indices_a) == expected_starts, f"Expected {expected_starts}, got {set(indices_a)}"
    
    # CHECK 3.2: NaN Checks
    # Create file C with observation but NaNs
    # Ticker C: Observable 10..20. NaNs at 15.
    # Window covering 15 should be dropped.
    # Window starts at S. Cover [S, S+7).
    # If S=10, covers 10..16 -> Includes 15. Dropped.
    # If S=8, covers 8..14 -> Safe.
    
    df_c = pl.DataFrame({
        "date": list(range(30)),
        "close": [100.0 if i != 15 else np.nan for i in range(30)]
    })
    file_c = tmp_path / "C.parquet"
    df_c.write_parquet(file_c)
    
    # Update mock to include C
    def mock_load_c(path):
        return {
            "A": {10,11,12,13,14,15,16,17,18,19,20}, 
            "B": set(),
            "C": {10,11,12,13,14,15,16,17,18,19,20} 
        }
    ChronosDataset._load_universe_mask = lambda self, p: mock_load_c(p)
    
    ds_c = ChronosDataset(
        files=[file_c],
        context_len=5,
        prediction_len=2,
        stride=2,
        target_col="close"
    )
    
    indices_c = [x[1] for x in ds_c.index]
    print(f"Indices C: {indices_c}")
    
    # 15 is NaN.
    # Window size 7 (5+2).
    # Starts that cover 15:
    # 15 in [S, S+7) => S <= 15 < S+7 => S > 8, S <= 15.
    # Range (9, 15].
    # Stride 2.
    # Potential starts: ... 8, 10, 12, 14, 16 ...
    # 10 covers 10..16 (includes 15) -> Drop
    # 12 covers 12..18 (includes 15) -> Drop
    # 14 covers 14..20 (includes 15) -> Drop
    
    # 8 covers 8..14 (Safe) -> Keep (Anchor 8+4=12 observable)
    # 16 covers 16..22 (Safe) -> Keep (Anchor 16+4=20 observable)
    
    assert 10 not in indices_c
    assert 12 not in indices_c
    assert 14 not in indices_c
    assert 8 in indices_c
    assert 16 in indices_c
    
    # CHECK 3.3: Scaling
    # Get item from A at start 6
    item = ds[0] # Should correspond to first index of A (6)
    
    # Data: 100 * 1.01^i
    # Ref index: 6 + 5 - 1 = 10.
    # Ref Val = 100 * 1.01^10
    # Val at 10 = Ref Val
    # Log(Val/Ref) = Log(1) = 0.
    
    past = item["past_target"]
    # Last element of past should be 0.
    assert torch.isclose(past[-1], torch.tensor(0.0), atol=1e-5)
    
    # Check shape
    assert past.shape == (5, 1)
    
    print("All checks passed.")

if __name__ == "__main__":
    pytest.main([__file__])
