"""Test PatchTST continuous output and RevIN denormalization.

Validates:
  1. ContinuousPatchTST produces continuous float output (not discrete tokens)
  2. MSE loss computes correctly
  3. RevIN denormalization preserves absolute magnitude on sine wave
  4. Output shapes match expected configuration
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from algaie.models.tsfm.patchtst import ContinuousPatchTST, RevIN


class TestRevIN:
    def test_normalize_denorm_roundtrip(self):
        """RevIN(denorm(norm(x))) ≈ x."""
        revin = RevIN(num_features=1, affine=False)
        x = torch.randn(4, 128)
        normed = revin(x, mode="norm")
        recovered = revin(normed, mode="denorm")
        torch.testing.assert_close(recovered, x, atol=1e-5, rtol=1e-5)

    def test_sine_wave_magnitude_preservation(self):
        """Deterministic sine wave through RevIN preserves absolute magnitude."""
        revin = RevIN(num_features=1, affine=False)
        t = torch.linspace(0, 4 * 3.14159, 256)
        x = (100.0 + 10.0 * torch.sin(t)).unsqueeze(0)  # (1, 256), around 100±10

        normed = revin(x, mode="norm")
        # Normalized should have ~zero mean, ~unit variance
        assert abs(normed.mean().item()) < 0.1
        assert abs(normed.std().item() - 1.0) < 0.2

        # Denorm should restore original magnitude
        recovered = revin(normed, mode="denorm")
        assert abs(recovered.mean().item() - 100.0) < 1.0
        torch.testing.assert_close(recovered, x, atol=1e-4, rtol=1e-4)





class TestContinuousPatchTST:
    @pytest.fixture
    def model(self):
        return ContinuousPatchTST(
            c_in=1, seq_len=128, patch_len=16, stride=8,
            forecast_horizon=1, d_model=64, n_heads=4, n_layers=2,
        )

    def test_output_is_continuous_float(self, model):
        """Output must be continuous float, not discrete integer tokens."""
        x = torch.randn(2, 128)
        out = model(x)
        assert out.dtype == torch.float32
        # Continuous: output should NOT be integers
        assert not torch.all(out == out.round()), "Output looks discrete — should be continuous float"

    def test_output_shape(self, model):
        """Shape matches (batch, forecast_horizon)."""
        x = torch.randn(4, 128)
        out = model(x)
        assert out.shape == (4, 1)

    def test_mse_loss(self, model):
        """MSE loss computes correctly and is differentiable."""
        x = torch.randn(4, 128)
        target = torch.randn(4, 1)
        pred = model(x)
        loss = ContinuousPatchTST.mse_loss(pred, target)
        assert loss.item() > 0
        assert loss.requires_grad

    def test_revin_denorm_restores_magnitude(self, model):
        """After full forward pass, output magnitude should relate to input magnitude."""
        # Use a strongly positive input (price ~4500)
        x = torch.ones(2, 128) * 4500.0 + torch.randn(2, 128) * 10.0
        out = model(x)
        # Output should be in a similar order of magnitude due to RevIN denorm
        assert out.abs().mean() > 1.0, "Output too small — RevIN denorm may have failed"

    def test_gradient_flow(self, model):
        """Gradients flow through entire network."""
        x = torch.randn(2, 128)
        target = torch.randn(2, 1)
        pred = model(x)
        loss = ContinuousPatchTST.mse_loss(pred, target)
        loss.backward()

        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"No gradient for {name}"
                assert not torch.all(p.grad == 0), f"Zero gradient for {name}"

    def test_multi_horizon(self):
        """Model works with forecast_horizon > 1."""
        model = ContinuousPatchTST(
            c_in=1, seq_len=128, patch_len=16, stride=8,
            forecast_horizon=5, d_model=64, n_heads=4, n_layers=2,
        )
        x = torch.randn(2, 128)
        out = model(x)
        assert out.shape == (2, 5)
