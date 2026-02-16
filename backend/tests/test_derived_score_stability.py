"""Tests for derived score stabilization (Deliverable B, updated for F2).

After F2, risk_pred is now raw log_sigma (not softplus'd sigma).
sigma = softplus(log_sigma) + sigma_floor, handled inside stabilize_derived_score.
"""
from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestStabilizeNumpy:
    def test_finite_output(self):
        from sleeves.cooc_reversal_futures.model.score_stabilizer import stabilize_derived_score

        raw = np.array([1.0, -2.0, 100.0, -100.0])
        # log_sigma values => sigma = softplus(log_sigma) + floor
        log_sigma = np.array([0.0, 0.01, 0.001, 10.0])
        out = stabilize_derived_score(raw, log_sigma)
        assert np.all(np.isfinite(out))

    def test_clamp_bounds(self):
        from sleeves.cooc_reversal_futures.model.score_stabilizer import stabilize_derived_score

        raw = np.array([1e6])
        log_sigma = np.array([0.001])
        out = stabilize_derived_score(raw, log_sigma, derived_clip=10.0)
        assert out[0] <= 10.0

    def test_sigma_floor_prevents_blowup(self):
        from sleeves.cooc_reversal_futures.model.score_stabilizer import stabilize_derived_score

        raw = np.array([1.0])
        # Very negative log_sigma => softplus ≈ 0, floor kicks in
        log_sigma = np.array([-100.0])
        out = stabilize_derived_score(raw, log_sigma, sigma_floor=0.05)
        assert np.isfinite(out[0])
        # sigma ≈ floor = 0.05, derived ≈ 1/(1e-6 + 0.05) ≈ 19.99
        assert out[0] < 25.0

    def test_tanh_squash(self):
        from sleeves.cooc_reversal_futures.model.score_stabilizer import stabilize_derived_score

        raw = np.array([100.0])
        # log_sigma = 0.0 => sigma = softplus(0) + floor ≈ 0.6931 + 1e-4 ≈ 0.6932
        log_sigma = np.array([0.0])
        out = stabilize_derived_score(raw, log_sigma, score_tanh=True)
        # tanh(100) ≈ 1.0, so derived = 1.0 / (1e-6 + ~0.6932) ≈ 1.443
        assert abs(out[0]) < 2.0  # bounded, tanh applied


@pytest.mark.skipif(not HAS_TORCH, reason="torch required")
class TestStabilizeTorch:
    def test_torch_matches_numpy(self):
        from sleeves.cooc_reversal_futures.model.score_stabilizer import stabilize_derived_score

        raw_np = np.array([0.5, -1.0, 3.0])
        log_sigma_np = np.array([0.1, 0.5, 2.0])
        out_np = stabilize_derived_score(raw_np, log_sigma_np)

        raw_t = torch.tensor(raw_np, dtype=torch.float32)
        log_sigma_t = torch.tensor(log_sigma_np, dtype=torch.float32)
        out_t = stabilize_derived_score(raw_t, log_sigma_t).numpy()

        np.testing.assert_allclose(out_np, out_t, rtol=1e-5)

    def test_torch_finite(self):
        from sleeves.cooc_reversal_futures.model.score_stabilizer import stabilize_derived_score

        raw = torch.tensor([1e6, -1e6, 0.0])
        log_sigma = torch.tensor([0.0, 0.0, 0.0])
        out = stabilize_derived_score(raw, log_sigma)
        assert torch.isfinite(out).all()


class TestManifestExtraction:
    def test_default_params(self):
        from sleeves.cooc_reversal_futures.model.score_stabilizer import stabilizer_params_from_manifest

        params = stabilizer_params_from_manifest({})
        assert params["sigma_floor"] == 1e-4
        assert params["sigma_cap"] is None
        assert params["score_tanh"] is False
        assert params["derived_clip"] == 10.0

    def test_custom_params(self):
        from sleeves.cooc_reversal_futures.model.score_stabilizer import stabilizer_params_from_manifest

        manifest = {
            "sigma_floor": 0.01,
            "sigma_cap": 5.0,
            "score_tanh": True,
            "derived_score_clip": 5.0,
        }
        params = stabilizer_params_from_manifest(manifest)
        assert params["sigma_floor"] == 0.01
        assert params["sigma_cap"] == 5.0
        assert params["score_tanh"] is True
        assert params["derived_clip"] == 5.0
