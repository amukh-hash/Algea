import pytest
import numpy as np
from app.features.signal_processing import apply_modwt_uks, sure_shrink, apply_sliding_wavelet_ukf

def test_sureshrink_preserves_jumps():
    """
    Verify that SureShrink preserves significant market jumps (alpha)
    better than aggressive thresholding would.
    """
    # Create a signal with a clear step change (jump)
    np.random.seed(42)
    n = 100
    jump_height = 10.0
    data = np.concatenate([np.zeros(n//2), np.ones(n//2) * jump_height])
    
    # Add some Gaussian noise
    noise = np.random.normal(0, 0.5, n)
    noisy_data = data + noise
    
    # Apply smoothing
    smoothed = apply_modwt_uks(noisy_data, level=2)
    
    # Check the jump amplitude in the smoothed signal
    # We look at the difference between the averages of the two halves
    # It shouldn't be too degraded
    
    pre_jump_mean = np.mean(smoothed[40:50])
    post_jump_mean = np.mean(smoothed[50:60])
    
    detected_jump = post_jump_mean - pre_jump_mean
    
    print(f"Original Jump: {jump_height}, Detected Jump: {detected_jump}")
    
    # We expect the jump to be largely preserved (e.g., > 75%)
    # VisuShrink often crushes it to < 50%
    assert detected_jump > jump_height * 0.75, f"Jump degraded too much: {detected_jump}"

def test_swt_padding_arbitrary_length():
    """
    Verify that apply_modwt_uks works for arbitrary data lengths
    that are NOT powers of 2, ensuring padding works.
    """
    # Length 101 is prime/odd, definitely not power of 2
    data = np.random.randn(101)
    
    try:
        smoothed = apply_modwt_uks(data, level=3)
        assert len(smoothed) == 101, "Output length mismatch"
        assert not np.any(np.isnan(smoothed)), "Output contains NaNs"
    except Exception as e:
        pytest.fail(f"apply_modwt_uks failed on arbitrary length: {e}")

def test_modwt_uks_short_signal():
    """Test on a very short signal to ensure it doesn't crash."""
    data = np.random.randn(10) # 10 < 2^3=8 is handled? 2^3=8. 
    # Logic: level=3 requires length >= 8 normally depending on wavelet implementation
    # But padding should handle it.
    
    try:
        smoothed = apply_modwt_uks(data, level=2) 
        # level 2 needs 4 samples roughly.
        assert len(smoothed) == 10
    except Exception as e:
        pytest.fail(f"Failed on short signal: {e}")

def test_sure_shrink_logic():
    """Test the sure_shrink function specifically."""
    # Case 1: Pure noise -> should shrink significantly
    noise = np.random.normal(0, 0.1, 1000)
    shrunk_noise = sure_shrink(noise)
    assert np.std(shrunk_noise) < np.std(noise), "Noise variance should decrease"
    
    # Case 2: Sparse large coeffs -> should be preserved
    coeffs = np.array([10.0, 0.1, 0.1, 0.0, 0.1])
    # The 10.0 is an outlier/signal, should be kept
    shrunk = sure_shrink(coeffs)
    assert np.abs(shrunk[0]) > 8.0, "Large coefficient should be preserved"
    # Small coeffs should be zeroed or reduced
    assert np.all(np.abs(shrunk[1:]) < 0.1), "Small coefficients should be shrunk"
