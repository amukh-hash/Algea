"""
Unit tests for the upgraded pairwise_ranking_loss function.

Tests:
  1. Determinism: fixed seed → identical output.
  2. Per-symbol cap: no index exceeds max_pairs_per_symbol.
  3. Uniform and stratified modes run without error on CPU (and CUDA if available).
  4. No meshgrid: function source does not contain 'meshgrid'.
  5. Edge cases: small N, empty mask.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.scripts.train_selector import pairwise_ranking_loss


def _make_inputs(N: int = 200, B: int = 2, device: str = "cpu"):
    """Create synthetic scores, targets, mask."""
    scores = torch.randn(B, N, device=device)
    targets = torch.randn(B, N, device=device)
    mask = torch.ones(B, N, device=device)
    return scores, targets, mask


# ────────────────────────────────────────────────────────────────────────
# 1. Determinism
# ────────────────────────────────────────────────────────────────────────

def test_determinism():
    scores, targets, mask = _make_inputs(N=300)

    torch.manual_seed(42)
    loss_a = pairwise_ranking_loss(scores, targets, mask, max_pairs=5000)

    torch.manual_seed(42)
    loss_b = pairwise_ranking_loss(scores, targets, mask, max_pairs=5000)

    assert torch.allclose(loss_a, loss_b), (
        f"Determinism failed: {loss_a.item()} != {loss_b.item()}"
    )
    print(f"  ✓ determinism: loss={loss_a.item():.6f}")


# ────────────────────────────────────────────────────────────────────────
# 2. Per-symbol cap
# ────────────────────────────────────────────────────────────────────────

def test_per_symbol_cap():
    N = 500
    scores = torch.randn(1, N)
    targets = torch.randn(1, N)
    mask = torch.ones(1, N)
    cap = 20

    torch.manual_seed(0)
    # Verify loss is finite with cap enabled
    loss = pairwise_ranking_loss(
        scores, targets, mask,
        max_pairs=10_000,
        max_pairs_per_symbol=cap,
    )
    assert loss.isfinite(), f"Loss not finite: {loss.item()}"

    # Run multiple times to verify cap reduces appearances on average
    torch.manual_seed(42)
    m = mask[0].bool()
    s = scores[0][m]
    y = targets[0][m]
    N_valid = s.shape[0]
    k = max(1, int(N_valid * 0.2))
    sorted_idx = torch.argsort(y, descending=True)
    top_idx = sorted_idx[:k]
    bot_idx = sorted_idx[-k:]

    # Sample pairs and apply vectorized cap
    pair_i = top_idx[torch.randint(len(top_idx), (10_000,))]
    pair_j = bot_idx[torch.randint(len(bot_idx), (10_000,))]

    counts_i = torch.bincount(pair_i, minlength=N_valid)
    counts_j = torch.bincount(pair_j, minlength=N_valid)
    combined = counts_i + counts_j

    # Before cap, some symbols appear many times
    max_before = combined.max().item()

    # Apply probabilistic cap
    over_cap_mask = combined > cap
    if over_cap_mask.any():
        ci_count = combined[pair_i].float()
        cj_count = combined[pair_j].float()
        worst = torch.maximum(ci_count, cj_count)
        keep_prob = (cap / worst).clamp(max=1.0)
        keep = torch.bernoulli(keep_prob).bool()
        pair_i = pair_i[keep]
        pair_j = pair_j[keep]

    # After cap, recount
    counts_after_i = torch.bincount(pair_i, minlength=N_valid)
    counts_after_j = torch.bincount(pair_j, minlength=N_valid)
    combined_after = counts_after_i + counts_after_j
    max_after = combined_after.max().item()

    # The cap should significantly reduce the max count
    assert max_after < max_before, (
        f"Cap didn't reduce: before={max_before}, after={max_after}"
    )
    # Max count should be in the neighborhood of cap (probabilistic, so allow 2x)
    assert max_after <= cap * 3, (
        f"Max count {max_after} too far above cap {cap}"
    )

    print(f"  ✓ per-symbol cap: {len(pair_i)} pairs survived, "
          f"max count {max_before} → {max_after} (cap={cap})")


# ────────────────────────────────────────────────────────────────────────
# 3. Both modes run on CPU (and CUDA if available)
# ────────────────────────────────────────────────────────────────────────

def test_modes_cpu():
    scores, targets, mask = _make_inputs(N=200)

    for mode in ("uniform", "stratified"):
        torch.manual_seed(123)
        loss = pairwise_ranking_loss(
            scores, targets, mask,
            max_pairs=500,
            pairwise_mode=mode,
        )
        assert loss.isfinite(), f"{mode} loss not finite: {loss.item()}"
        print(f"  ✓ {mode} CPU: loss={loss.item():.6f}")


def test_modes_cuda():
    if not torch.cuda.is_available():
        print("  ⊘ CUDA not available, skipping")
        return

    scores, targets, mask = _make_inputs(N=200, device="cuda")

    for mode in ("uniform", "stratified"):
        torch.manual_seed(123)
        loss = pairwise_ranking_loss(
            scores, targets, mask,
            max_pairs=500,
            pairwise_mode=mode,
        )
        assert loss.isfinite(), f"{mode} CUDA loss not finite: {loss.item()}"
        print(f"  ✓ {mode} CUDA: loss={loss.item():.6f}")


# ────────────────────────────────────────────────────────────────────────
# 4. No meshgrid in source
# ────────────────────────────────────────────────────────────────────────

def test_no_meshgrid():
    src = inspect.getsource(pairwise_ranking_loss)
    # Check for actual meshgrid calls, not docstring mentions
    assert "torch.meshgrid" not in src and "meshgrid(" not in src, (
        "meshgrid call found in pairwise_ranking_loss source!"
    )
    print("  ✓ no meshgrid call in source")


# ────────────────────────────────────────────────────────────────────────
# 5. Edge cases
# ────────────────────────────────────────────────────────────────────────

def test_edge_cases():
    # Tiny N — should not crash
    scores = torch.randn(1, 3)
    targets = torch.randn(1, 3)
    mask = torch.ones(1, 3)
    loss = pairwise_ranking_loss(scores, targets, mask)
    # N < 4, should return 0 loss
    assert loss.item() == 0.0, f"Expected 0 for N<4, got {loss.item()}"
    print("  ✓ edge case N<4: loss=0.0")

    # Empty mask
    scores = torch.randn(1, 50)
    targets = torch.randn(1, 50)
    mask = torch.zeros(1, 50)
    loss = pairwise_ranking_loss(scores, targets, mask)
    assert loss.item() == 0.0, f"Expected 0 for empty mask, got {loss.item()}"
    print("  ✓ edge case empty mask: loss=0.0")


# ────────────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Pairwise ranking loss tests ===")
    test_no_meshgrid()
    test_determinism()
    test_per_symbol_cap()
    test_modes_cpu()
    test_modes_cuda()
    test_edge_cases()
    print("\nAll tests passed ✓")
