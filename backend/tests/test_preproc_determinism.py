import pytest
import polars as pl
import numpy as np
import os
import json
from backend.app.preprocessing import preproc

def test_preproc_determinism(tmp_path):
    # Create dummy data
    df = pl.DataFrame({
        "timestamp": [1, 2, 3], # placeholder
        "close": [10.0, 20.0, 40.0],
        "volume": [100.0, 100.0, 100.0], # log1p(100) ~ 4.615. Mean=4.615, Std=0 -> NaN/Inf issue?
        # Variance of volume is 0 -> Std is 0 -> Zscore is NaN/Inf.
        # We should use data with variance.
        "ad_line": [1.0, 2.0, 3.0],
        "bpi": [10.0, 20.0, 30.0]
    })
    
    # Adjust volume to have variance
    df = df.with_columns(pl.Series("volume", [10, 100, 1000]))
    
    p = preproc.Preprocessor()
    p.fit(df)
    
    # Transform
    res1 = p.transform(df)
    
    # Save/Load
    save_path = tmp_path / "preproc.json"
    p.save(str(save_path))
    
    p2 = preproc.Preprocessor.load(str(save_path))
    res2 = p2.transform(df)
    
    # Check equality
    assert res1.equals(res2)
    
    # Check specific calculation
    # AD Line: 1, 2, 3 -> Mean 2, Std 1 (sample std? polars default is sample - divisor n-1)
    # (1-2)/1 = -1
    # (2-2)/1 = 0
    # (3-2)/1 = 1
    ad_vals = res2["ad_line_norm"].to_list()
    assert np.allclose(ad_vals, [-1.0, 0.0, 1.0])

def test_preproc_tamper_protection(tmp_path):
    df = pl.DataFrame({
        "timestamp": [1],
        "close": [10.0],
        "volume": [10.0],
        "ad_line": [1.0],
        "bpi": [1.0]
    })
    
    p = preproc.Preprocessor()
    p.fit(df)
    save_path = tmp_path / "preproc_tampered.json"
    p.save(str(save_path))
    
    # Tamper with file
    with open(save_path, 'r') as f:
        data = json.load(f)
    
    # Change a param
    data["params"]["ad_line"]["mean"] += 0.1
    
    with open(save_path, 'w') as f:
        json.dump(data, f)
        
    # Load should fail
    with pytest.raises(ValueError, match="Preprocessor hash mismatch"):
        preproc.Preprocessor.load(str(save_path))
