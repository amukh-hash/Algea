"""
Test: Inference determinism.

Validates that ``.eval()`` mode is properly enforced — repeated
forward passes with the same input must produce identical outputs
(zero variance), proving dropout layers are disabled.
"""
from __future__ import annotations

import torch
import pytest


class TestITransformerDeterminism:
    """iTransformer must produce deterministic output in eval mode."""

    def test_repeated_forward_zero_variance(self):
        from algae.models.tsfm.itransformer import iTransformer

        model = iTransformer(
            num_variates=6,
            lookback_len=60,
            pred_len=5,
            d_model=64,
            n_heads=4,
            e_layers=2,
            dropout=0.1,
        )
        model.eval()

        torch.manual_seed(42)
        x = torch.randn(1, 60, 6)

        outputs = []
        for _ in range(100):
            with torch.inference_mode():
                out = model(x)
            outputs.append(out.clone())

        stacked = torch.stack(outputs)
        variance = stacked.var(dim=0).max().item()
        assert variance == 0.0, (
            f"Repeated eval-mode forward passes must produce zero variance.  "
            f"Got max variance = {variance:.2e}.  Dropout may be leaking."
        )

    def test_input_shape_validation(self):
        """iTransformer must reject mismatched input shapes."""
        from algae.models.tsfm.itransformer import iTransformer

        model = iTransformer(num_variates=6, lookback_len=60, pred_len=5)
        model.eval()

        # Wrong time dimension
        with pytest.raises(ValueError, match="Expected shape"):
            model(torch.randn(1, 30, 6))

        # Wrong variate dimension
        with pytest.raises(ValueError, match="Expected shape"):
            model(torch.randn(1, 60, 10))

    def test_output_shape(self):
        """iTransformer output must be [B, Variates, Horizon]."""
        from algae.models.tsfm.itransformer import iTransformer

        model = iTransformer(num_variates=6, lookback_len=60, pred_len=5)
        model.eval()

        x = torch.randn(2, 60, 6)
        with torch.inference_mode():
            out = model(x)

        assert out.shape == (2, 6, 5), f"Expected (2, 6, 5), got {out.shape}"


class TestRankTransformerDeterminism:
    """RankTransformer output must be deterministic in eval mode."""

    def test_repeated_forward_zero_variance(self):
        from algae.models.ranker.rank_transformer import RankTransformer

        model = RankTransformer(d_input=18, d_model=64, n_head=4, n_layers=2)
        model.eval()

        torch.manual_seed(42)
        x = torch.randn(1, 20, 18)

        outputs = []
        for _ in range(50):
            with torch.inference_mode():
                out = model(x)
            scores = out["score"]
            outputs.append(scores.clone())

        stacked = torch.stack(outputs)
        variance = stacked.var(dim=0).max().item()
        assert variance == 0.0, (
            f"RankTransformer eval-mode variance must be 0.0, got {variance:.2e}"
        )
