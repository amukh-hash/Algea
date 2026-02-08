
import pytest
import pandas as pd
import polars as pl
import numpy as np
import os
import shutil
import subprocess
from pathlib import Path
from backend.app.data import priors_store

def test_feature_merge_priors_integration(tmp_path):
    """
    Integration test for features_partition_featureframe.py
    Verifies that priors are correctly merged into the feature frame before partitioning.
    """
    # 1. Setup Paths
    features_dir = tmp_path / "features"
    features_dir.mkdir()
    priors_dir = tmp_path / "priors"
    priors_dir.mkdir()
    out_dir = tmp_path / "output"
    out_dir.mkdir()
    
    feature_path = features_dir / "featureframe_vv1.parquet"
    
    # 2. Create Dummy FeatureFrame
    dates = pd.date_range("2023-01-01", "2023-01-05")
    tickers = ["AAPL", "GOOGL"]
    
    data = []
    for t in tickers:
        for d in dates:
            data.append({
                "date": d,
                "symbol": t,
                "close": 100.0,
                "volume": 1000
            })
    
    df_features = pd.DataFrame(data)
    df_features.to_parquet(feature_path)
    
    # 3. Create Dummy PriorsFrame
    # We use priors_store to write it to ensure format matches
    # Priors for AAPL only, and only for some dates
    priors_data = []
    for d in dates[:3]: # First 3 days
        priors_data.append({
            "date": d.date(), # Polars expects date objects usually, or string
            "ticker": "AAPL",
            "p_mu5": 0.01,
            "p_mu10": 0.02,
            "p_sig5": 0.05,
            "p_sig10": 0.06,
            "p_pdown5": 0.4,
            "p_pdown10": 0.45
        })
        
    df_priors = pl.DataFrame(priors_data)
    
    meta = {"test": True}
    priors_store.write_priors_frame(df_priors, str(priors_dir), meta)
    
    # 4. Run Script
    script_path = list(Path("backend/scripts/features").glob("features_partition_featureframe.py"))[0]
    # Assume we are in root
    cmd = [
        "python", str(script_path),
        "--featureframe_path", str(feature_path),
        "--priors_dir", str(priors_dir),
        "--output_dir", str(out_dir)
    ]
    
    # Set env var to ensure config.PRIORS_ENABLED is True (it is default True in code, but good to be safe)
    env = os.environ.copy()
    
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    
    if result.returncode != 0:
        print("Script STDOUT:", result.stdout)
        print("Script STDERR:", result.stderr)
        
    assert result.returncode == 0, "Script failed"
    
    # 5. Verify Output
    # Check AAPL.parquet
    aapl_path = out_dir / "AAPL.parquet"
    assert aapl_path.exists()
    
    df_aapl = pd.read_parquet(aapl_path)
    
    # Expected: 5 rows
    assert len(df_aapl) == 5
    
    # Check Priors columns exist
    assert "p_mu5" in df_aapl.columns
    assert "p_mu5_isna" in df_aapl.columns
    
    # Check values
    # First 3 rows should have values
    valid_rows = df_aapl.iloc[:3]
    # float comparison
    assert np.allclose(valid_rows["p_mu5"].values, 0.01)
    
    # Last 2 rows should be NaN or 0 if we filled?
    # Logic in script: `df = pd.merge(..., how="left")`.
    # And we added `_isna` col.
    # We didn't fill NaNs in the script (commented out "pass").
    # So they should be NaN.
    
    missing_rows = df_aapl.iloc[3:]
    assert missing_rows["p_mu5"].isna().all()
    assert (missing_rows["p_mu5_isna"] == 1).all()
    
    # Check GOOGL
    googl_path = out_dir / "GOOGL.parquet"
    assert googl_path.exists()
    df_googl = pd.read_parquet(googl_path)
    # merged left, so all rows present
    assert len(df_googl) == 5
    # All priors should be NaN
    assert df_googl["p_mu5"].isna().all()
    assert (df_googl["p_mu5_isna"] == 1).all()

if __name__ == "__main__":
    # Manually run if executed as script
    import sys
    sys.exit(pytest.main(["-v", __file__]))
