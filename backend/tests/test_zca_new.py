import pytest
import numpy as np
from app.preprocessing.zca import zca_whitening, apply_zca_transform

def test_zca_decorrelates():
    """Verify ZCA whitening produces diagonal covariance matrix."""
    # Create correlated data
    # x = rand, y = x + noise
    n_samples = 1000
    x = np.random.randn(n_samples)
    y = x * 0.8 + np.random.randn(n_samples) * 0.2
    
    data = np.stack([x, y], axis=1) # (1000, 2)
    
    # Check initial correlation
    cov_orig = np.cov(data, rowvar=False)
    assert abs(cov_orig[0, 1]) > 0.5 # Should be correlated
    
    # Apply ZCA
    data_whitened, zca_mat = zca_whitening(data)
    
    # Check whitened covariance
    cov_white = np.cov(data_whitened, rowvar=False)
    
    # Off-diagonals should be close to 0
    assert abs(cov_white[0, 1]) < 0.1, "Off-diagonal covariance should be near zero"
    
    # Diagonals should be close to 1 (unit variance) is typically Sphering.
    # ZCA preserves original variances sum? Or scales to unit?
    # Our implementation: U * 1/sqrt(S) * U.T
    # This spheres the data (Cov = I).
    assert abs(cov_white[0, 0] - 1.0) < 0.1
    assert abs(cov_white[1, 1] - 1.0) < 0.1

def test_apply_zca():
    """Verify applying the matrix works."""
    data = np.random.randn(10, 5)
    data_w, mat = zca_whitening(data)
    
    data_w_2 = apply_zca_transform(data - np.mean(data, axis=0), mat)
    
    # Should be identical
    assert np.allclose(data_w, data_w_2)
