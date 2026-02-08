
import pytest
import numpy as np
import torch
from pathlib import Path
from backend.app.models.selector_scaler import SelectorFeatureScaler

def test_selector_scaler_fit_transform_separation():
    """
    # Verify that fitting on Train does not see Test data,
    # and transforming Test uses Train stats.
    """
    np.random.seed(42)
    scaler = SelectorFeatureScaler()
    
    # Train data: Mean=10, Spread small
    train_data = np.random.normal(10.0, 1.0, (100, 5))
    
    # Test data: Mean=100, Spread large (Shifted Regime)
    test_data = np.random.normal(100.0, 5.0, (100, 5))
    
    # Fit on Train
    scaler.fit(train_data)
    
    # Check internals (RobustScaler stores center/scale)
    # Sklearn RobustScaler: center_ = median
    medians = scaler.scaler.center_
    assert np.allclose(medians, 10.0, atol=0.5), f"Medians should be ~10, got {medians}"
    
    # Transform Train -> Should be approx 0 median
    valid_train = scaler.transform(train_data)
    assert np.allclose(np.median(valid_train, axis=0), 0.0, atol=0.5)
    
    # Transform Test -> Should be huge (since Test is ~100 and Train was ~10)
    # (100 - 10) / scale
    valid_test = scaler.transform(test_data)
    assert np.all(valid_test > 10.0), "Test data should be scaled relative to Train stats (large positive)"
    
    # Ensure Test data did NOT affect fit
    # Refit on Test
    scaler2 = SelectorFeatureScaler()
    scaler2.fit(test_data)
    assert np.allclose(scaler2.scaler.center_, 100.0, atol=1.0)
    
    # The first scaler should still have old medians
    assert np.allclose(scaler.scaler.center_, 10.0, atol=0.5)

def test_selector_scaler_serialization(tmp_path):
    """
    Verify save/load works and preserves stats.
    """
    path = tmp_path / "scaler.joblib"
    
    scaler = SelectorFeatureScaler(version="v2")
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]) # Medians: 3.0, 4.0
    scaler.fit(data, feature_names=["A", "B"])
    
    scaler.save(str(path))
    
    loaded = SelectorFeatureScaler.load(str(path))
    
    assert loaded.version == "v2"
    assert loaded.feature_names == ["A", "B"]
    assert loaded.fitted is True
    assert np.allclose(loaded.scaler.center_, scaler.scaler.center_)
    
    # Transform check
    t1 = scaler.transform(data)
    t2 = loaded.transform(data)
    assert np.allclose(t1, t2)

if __name__ == "__main__":
    test_selector_scaler_fit_transform_separation()
    test_selector_scaler_serialization(Path("."))
    print("ALL TESTS PASSED")
