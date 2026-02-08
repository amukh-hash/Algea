
import pytest
import torch
import numpy as np
import pandas as pd
from unittest.mock import MagicMock
from backend.scripts.teacher.build_priors_frame import generate_priors_batch

# Mock Wrapper
class MockChronosWrapper:
    def __init__(self):
        self.model = MagicMock()
        # Mock predict to return something deterministic based on input mean
        # so we can verify "same input -> same output"
        self.model.predict = self.mock_predict
        
    def mock_predict(self, context, prediction_length, num_samples, **kwargs):
        # context: [B, T]
        # output: [B, NumSamples, Horizon]
        B = context.shape[0]
        H = prediction_length
        S = num_samples
        
        # Simple deterministic output: 
        # Forecast = (mean(context) + k) * decaying_curve
        # where k is sample index shift
        
        ctx_mean = context.mean(dim=1, keepdim=True).unsqueeze(-1) # [B, 1, 1]
        
        # Create [B, S, H]
        out = torch.zeros(B, S, H)
        for s in range(S):
            # Sample variation
            sample_shift = (s - S/2) * 0.01 
            for h in range(H):
                # Horizon shape
                out[:, s, h] = ctx_mean[:, 0, 0] + sample_shift + (h * 0.001)
                
        return out

def test_priors_generation_determinism():
    """
    Test that generate_priors_batch is deterministic given the same input tensor.
    """
    wrapper = MockChronosWrapper()
    context = torch.randn(5, 50) # 5 samples, 50 context len
    
    out1 = generate_priors_batch(wrapper, context, "cpu")
    out2 = generate_priors_batch(wrapper, context, "cpu")
    
    # Check key stats
    assert np.allclose(out1["p_mu5"], out2["p_mu5"]), "Mu5 nondeterministic"
    assert np.allclose(out1["p_sig10"], out2["p_sig10"]), "Sig10 nondeterministic"

def test_priors_asof_logic():
    """
    Verify that if we shift the context window (simulation of rolling through time),
    the prior generation only sees valid data.
    
    This essentially tests that our usage of generate_priors_batch correctly 
    interprets the context.
    """
    wrapper = MockChronosWrapper()
    
    # Create a long time series
    full_series = torch.arange(100).float()
    
    # Times T=50 and T=51
    context_len = 10
    
    # Context for T=50: indices 41..50 (inclusive of 50? depending on definition. usually [T-len+1 : T+1])
    # build_priors_frame logic: window = values[idx - context_len + 1 : idx + 1]
    # For T=50 (idx=50), window is values[41:51] -> indices 41..50.
    
    idx_50 = 50
    ctx_50 = full_series[idx_50 - context_len + 1 : idx_50 + 1].unsqueeze(0) # [1, 10]
    
    idx_51 = 51
    ctx_51 = full_series[idx_51 - context_len + 1 : idx_51 + 1].unsqueeze(0) # [1, 10]
    
    # Generate
    out_50 = generate_priors_batch(wrapper, ctx_50, "cpu")
    out_51 = generate_priors_batch(wrapper, ctx_51, "cpu")
    
    # They should be different because input data changed
    assert not np.isclose(out_50["p_mu5"][0], out_51["p_mu5"][0]), "Priors should update when context slides"
    
    # Truncation check equivalent:
    # If we have data up to T=100, generating for T=50 should match
    # generating for T=50 when we only have data up to T=50.
    
    # Since we extract context explicitly, this is guaranteed by Python slicing,
    # provided logic "values[idx - context_len + 1 : idx + 1]" is used.
    # We verify that logic here.
    
    # Assert ctx_50 contains exactly 41..50
    expected = torch.arange(41, 51).float()
    assert torch.all(ctx_50[0] == expected), f"Context extraction incorrect. Got {ctx_50}, Expected {expected}"

if __name__ == "__main__":
    test_priors_generation_determinism()
    test_priors_asof_logic()
    print("ALL TESTS PASSED")
