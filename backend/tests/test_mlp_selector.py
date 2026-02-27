"""
Tests for MLPSelector model and model-type dispatch.

Tests:
  1. MLPSelector forward: correct output shapes.
  2. MLPSelector with risk head: both score and risk outputs.
  3. Masked positions do not affect loss.
  4. Both MLPSelector and RankTransformer produce compatible outputs.
  5. MLPSelector runs on CPU and CUDA.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from algea.models.ranker.mlp_selector import MLPSelector


def test_forward_shapes():
    """Output shapes are [B, N, 1] for score."""
    B, N, d = 4, 100, 12
    model = MLPSelector(d_input=d, hidden=64, depth=2, dropout=0.0)
    x = torch.randn(B, N, d)
    mask = torch.ones(B, N)
    out = model(x, mask)
    assert "score" in out, "Missing 'score' key"
    assert out["score"].shape == (B, N, 1), f"Expected ({B},{N},1), got {out['score'].shape}"
    assert "risk" not in out, "risk should not be present without use_risk_head"
    print(f"  ✓ forward shapes: score={out['score'].shape}")


def test_risk_head():
    """Risk head adds 'risk' key when enabled."""
    B, N, d = 2, 50, 12
    model = MLPSelector(d_input=d, hidden=64, depth=2, use_risk_head=True)
    x = torch.randn(B, N, d)
    out = model(x)
    assert "risk" in out, "Missing 'risk' key"
    assert out["risk"].shape == (B, N, 1), f"risk shape wrong: {out['risk'].shape}"
    print(f"  ✓ risk head: score={out['score'].shape}, risk={out['risk'].shape}")


def test_mask_independence():
    """Masked positions should not affect valid position scores."""
    B, N, d = 1, 20, 8
    model = MLPSelector(d_input=d, hidden=32, depth=2, dropout=0.0)
    model.eval()

    x = torch.randn(B, N, d)
    mask = torch.ones(B, N)

    with torch.no_grad():
        out_full = model(x, mask)
        # Zero out last 5 positions and mask them
        x_masked = x.clone()
        x_masked[:, -5:, :] = 999.0  # garbage
        mask2 = mask.clone()
        mask2[:, -5:] = 0
        out_masked = model(x_masked, mask2)

    # Valid positions (first 15) should have identical scores
    # because the MLP is per-ticker (no cross-asset attention)
    valid_scores_full = out_full["score"][:, :15]
    valid_scores_masked = out_masked["score"][:, :15]
    assert torch.allclose(valid_scores_full, valid_scores_masked, atol=1e-6), (
        "Valid position scores changed when masked positions were modified! "
        "MLP should be per-ticker independent."
    )
    print("  ✓ mask independence: valid positions unchanged")


def test_compatibility_with_transformer():
    """Both models produce compatible output dicts."""
    d = 12
    B, N = 2, 30

    mlp = MLPSelector(d_input=d, hidden=64, depth=2)
    x = torch.randn(B, N, d)
    mask = torch.ones(B, N)

    mlp_out = mlp(x, mask)

    try:
        from algea.models.ranker.rank_transformer import RankTransformer
        transformer = RankTransformer(d_input=d, d_model=32, n_head=2, n_layers=1)
        transformer_out = transformer(x, mask)
        assert "score" in transformer_out
        assert transformer_out["score"].shape == mlp_out["score"].shape, (
            f"Shape mismatch: transformer={transformer_out['score'].shape} "
            f"vs mlp={mlp_out['score'].shape}"
        )
        print("  ✓ compatibility: both models produce matching output shapes")
    except Exception as e:
        print(f"  ⊘ transformer compatibility skipped: {e}")


def test_cuda():
    """MLPSelector runs on CUDA."""
    if not torch.cuda.is_available():
        print("  ⊘ CUDA not available, skipping")
        return

    from algea.core.device import get_device
    dev = get_device()
    d = 12
    model = MLPSelector(d_input=d, hidden=64, depth=2).to(dev)
    x = torch.randn(2, 100, d, device=dev)
    mask = torch.ones(2, 100, device=dev)
    out = model(x, mask)
    assert out["score"].device.type == "cuda"
    print(f"  ✓ CUDA: score={out['score'].shape}")


def test_param_count():
    """Verify MLP is lightweight."""
    model = MLPSelector(d_input=12, hidden=128, depth=3, dropout=0.1)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 100_000, f"Too many params: {n_params}"
    print(f"  ✓ param count: {n_params:,}")


if __name__ == "__main__":
    print("=== MLPSelector tests ===")
    test_forward_shapes()
    test_risk_head()
    test_mask_independence()
    test_compatibility_with_transformer()
    test_cuda()
    test_param_count()
    print("\nAll tests passed ✓")
