"""
Test Weighted Pairwise Loss
Verifies determinism, correctness, and gradient flow.
"""
import pytest
import torch
from backend.app.models.selector_v2 import WeightedPairwiseLoss

def test_pairwise_loss_determinism():
    """
    Verify that loss is identical for same input and seed, 
    ensuring pair sampling is deterministic.
    """
    seed = 12345
    loss_fn = WeightedPairwiseLoss(seed=seed, max_pairs=10)
    
    B, N = 2, 100
    scores = torch.randn(B, N, requires_grad=True)
    p_trade = torch.rand(B, N, requires_grad=True)
    y_rank = torch.randn(B, N) # random ranks
    y_trade = torch.randint(0, 2, (B, N)).float()
    w = torch.ones(B, N)
    mask = torch.ones(B, N, dtype=torch.bool)
    
    # Run 1
    l1, _ = loss_fn(scores, p_trade, y_rank, y_trade, w, mask)
    
    # Run 2 (Re-instantiate or same instance? Instance resets gen with manual seed?)
    # forward() in my implementation does:
    # rng = torch.Generator(device=scores.device)
    # rng.manual_seed(self.seed)
    # So using same instance should be deterministic.
    
    l2, _ = loss_fn(scores, p_trade, y_rank, y_trade, w, mask)
    
    assert torch.isclose(l1, l2), f"Loss mismatch: {l1} != {l2}"
    
    # Verify randomness of sampling by changing seed
    loss_fn_diff = WeightedPairwiseLoss(seed=seed+1, max_pairs=10)
    l3, _ = loss_fn_diff(scores, p_trade, y_rank, y_trade, w, mask)
    
    # It MIGHT be different if max_pairs < total pairs.
    # With N=100, Quantile=0.2 => Top 20, Bottom 20. Total pairs 400.
    # max_pairs=10 forces sampling.
    assert not torch.isclose(l1, l3), "Loss should differ with different seed"

def test_gradient_flow():
    """Verify gradients propagate to scores and p_trade"""
    loss_fn = WeightedPairwiseLoss()
    B, N = 2, 50
    scores = torch.randn(B, N, requires_grad=True)
    p_trade = torch.rand(B, N, requires_grad=True)
    y_rank = torch.randn(B, N)
    y_trade = torch.randint(0, 2, (B, N)).float()
    w = torch.ones(B, N)
    mask = torch.ones(B, N, dtype=torch.bool)
    
    loss, metrics = loss_fn(scores, p_trade, y_rank, y_trade, w, mask)
    
    loss.backward()
    
    assert scores.grad is not None
    assert p_trade.grad is not None
    assert scores.grad.abs().sum() > 0
    assert p_trade.grad.abs().sum() > 0
    
    print(f"Loss: {loss.item()}, Rank: {metrics['loss_rank']}, Trade: {metrics['loss_trade']}")

if __name__ == "__main__":
    pytest.main([__file__])
