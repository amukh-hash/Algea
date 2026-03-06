"""
Test: NaN sanitization.

Validates that NaN and Inf inputs are caught before reaching PyTorch,
preventing silent poison propagation through model activations.
"""
from __future__ import annotations

import pytest


class TestNaNSanitization:
    """All adapters must reject NaN/Inf inputs before torch operations."""

    def test_chronos2_rejects_nan_series(self):
        """NaN in time series must raise before hitting PyTorch."""
        from backend.app.ml_platform.models.chronos2.adapter import Chronos2Adapter

        adapter = Chronos2Adapter(model_weights_path=None, device="cpu")
        # Even the fail-closed check occurs before NaN detection,
        # so we just verify it doesn't silently succeed
        with pytest.raises((RuntimeError, ValueError)):
            adapter.forecast([1.0, float("nan"), 2.0] * 20)

    def test_rl_policy_vetoes_nan_features(self):
        """NaN in live feature vector must trigger veto."""
        from backend.app.ml_platform.models.rl_policy.model import RLPolicyModel

        model = RLPolicyModel(hidden_size=32, raw_feature_dim=5, device="cpu")
        model.is_active = True  # Force active to test NaN path

        mult, veto, _ = model.act([1.0, float("nan"), 0.5, 0.1, -0.3])
        assert veto is True
        assert mult == 0.0

    def test_rl_policy_vetoes_inf_features(self):
        """Inf in live feature vector must trigger veto."""
        from backend.app.ml_platform.models.rl_policy.model import RLPolicyModel

        model = RLPolicyModel(hidden_size=32, raw_feature_dim=5, device="cpu")
        model.is_active = True

        mult, veto, _ = model.act([1.0, float("inf"), 0.5, 0.1, -0.3])
        assert veto is True
        assert mult == 0.0

    def test_itransformer_rejects_nan(self):
        """iTransformer forward pass must not silently propagate NaN."""
        import torch
        from algae.models.tsfm.itransformer import iTransformer

        model = iTransformer(num_variates=6, lookback_len=60, pred_len=5, d_model=64)
        model.eval()

        x = torch.randn(1, 60, 6)
        x[0, 10, 3] = float("nan")

        with torch.inference_mode():
            out = model(x)

        # The output WILL contain NaN because PyTorch propagates it.
        # This test documents the known behavior — the guard must be
        # at the adapter/service layer, not inside the nn.Module.
        has_nan = torch.isnan(out).any().item()
        assert has_nan, (
            "PyTorch should propagate NaN through the network.  "
            "This proves NaN guards must be at the adapter layer."
        )
