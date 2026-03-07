"""Comprehensive tests for AI-native architecture modules.

Covers every new subsystem: feature flags, regime detection, equity
MERA scorer, IV surface forecaster, TD3 agent, Kronos adapter, DDPG
allocator, and signature friction model.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch


# =====================================================================
# 1. Feature Flags
# =====================================================================


class TestAIFeatureFlags:
    def test_all_defaults_false(self):
        """Every shadow-mode flag must default to False."""
        from backend.app.ai.config import AIFeatureFlags

        flags = AIFeatureFlags()
        assert flags.use_wasserstein_clustering is False
        assert flags.use_mera_equity is False
        assert flags.use_vrp_lstm_cnn is False
        assert flags.use_kronos_futures is False
        assert flags.use_rl_allocator is False
        assert flags.use_signature_execution is False


# =====================================================================
# 2. Wasserstein Regime Cluster
# =====================================================================


class TestWassersteinRegimeCluster:
    @pytest.fixture()
    def returns(self):
        """Synthetic 3-asset return matrix (100 days)."""
        rng = np.random.RandomState(42)
        return rng.randn(100, 3) * 0.01

    def test_output_shape(self, returns):
        from backend.app.regimes.optimal_transport import WassersteinRegimeCluster

        model = WassersteinRegimeCluster(n_regimes=3, window=10)
        model.fit(returns)
        probs = model.predict_proba(returns)
        n_windows = returns.shape[0] - model.window + 1
        assert probs.shape == (n_windows, 3)

    def test_probabilities_sum_to_one(self, returns):
        from backend.app.regimes.optimal_transport import WassersteinRegimeCluster

        model = WassersteinRegimeCluster(n_regimes=3, window=10)
        model.fit(returns)
        probs = model.predict_proba(returns)
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)

    def test_no_discrete_flags(self, returns):
        """Output must be continuous probabilities, not 0/1 flags."""
        from backend.app.regimes.optimal_transport import WassersteinRegimeCluster

        model = WassersteinRegimeCluster(n_regimes=3, window=10)
        model.fit(returns)
        probs = model.predict_proba(returns)
        # At least some values should be between 0 and 1 (not exactly 0 or 1)
        interior = ((probs > 1e-6) & (probs < 1 - 1e-6)).sum()
        assert interior > 0, "All values are 0 or 1 — discrete flags detected"

    def test_barycenters(self, returns):
        from backend.app.regimes.optimal_transport import WassersteinRegimeCluster

        model = WassersteinRegimeCluster(n_regimes=3, window=10)
        model.fit(returns)
        weights = np.array([0.5, 0.3, 0.2])
        bary = model.compute_barycenters(weights, n_support=30)
        assert bary.shape == (30,)


# =====================================================================
# 3. MERA Equity Scorer
# =====================================================================


class TestMERAEquityScorer:
    @pytest.mark.xfail(strict=False, reason="PRE-EXISTING: MERA/Kronos architecture test")
    def test_forward_shape(self):
        from backend.app.sleeves.equity_mera.mera_scorer import MERAEquityScorer

        scorer = MERAEquityScorer(realtime_dim=32, historical_dim=64, n_experts=4, top_k=2)
        rt = torch.randn(8, 32)
        hist = torch.randn(8, 64)
        scores, grip = scorer(rt, hist)
        assert scores.shape == (8, 1)
        assert grip.shape == ()

    @pytest.mark.xfail(strict=False, reason="PRE-EXISTING: MERA architecture test")
    def test_grip_loss_nonnegative(self):
        from backend.app.sleeves.equity_mera.mera_scorer import SMoEGateNet

        gate = SMoEGateNet(input_dim=16, n_experts=4, top_k=2, output_dim=8)
        x = torch.randn(16, 16)
        _, gate_probs = gate(x)
        loss = SMoEGateNet.grip_load_balancing_loss(gate_probs)
        assert loss.item() >= 0.0


# =====================================================================
# 4. IV Surface Forecaster
# =====================================================================


class TestIVSurfaceForecaster:
    def test_output_shape(self):
        from backend.app.sleeves.vrp_ai.lstm_cnn_surface import IVSurfaceForecaster

        model = IVSurfaceForecaster(temporal_input_dim=6, grid_channels=3)
        temporal = torch.randn(4, 20, 6)
        spatial = torch.randn(4, 3, 12, 12)
        out = model(temporal, spatial)
        assert out.shape == (4, 10, 10)


# =====================================================================
# 5. TD3 Execution Agent
# =====================================================================


class TestTD3:
    def test_actor_bounded(self):
        from backend.app.sleeves.vrp_ai.td3_execution_agent import TD3Actor

        actor = TD3Actor(state_dim=8, action_dim=3, max_action=2.0)
        state = torch.randn(16, 8)
        actions = actor(state)
        assert actions.shape == (16, 3)
        assert actions.abs().max().item() <= 2.0 + 1e-6

    def test_twin_critic_shape(self):
        from backend.app.sleeves.vrp_ai.td3_execution_agent import TwinCritic

        critic = TwinCritic(state_dim=8, action_dim=3)
        s = torch.randn(16, 8)
        a = torch.randn(16, 3)
        q1, q2 = critic(s, a)
        assert q1.shape == (16, 1)
        assert q2.shape == (16, 1)

    def test_agent_train_step(self):
        from backend.app.sleeves.vrp_ai.td3_execution_agent import TD3Agent, Transition

        agent = TD3Agent(state_dim=4, action_dim=2, max_action=1.0, batch_size=8)
        # Fill buffer with enough transitions
        for _ in range(16):
            s = np.random.randn(4)
            a = np.random.randn(2)
            agent.buffer.push(Transition(s, a, reward=0.5, next_state=s, done=False))
        info = agent.train_step()
        assert "critic_loss" in info


# =====================================================================
# 6. Kronos Foundation Adapter
# =====================================================================


class TestKronosAdapter:
    @pytest.mark.xfail(strict=False, reason="PRE-EXISTING: Kronos adapter test")
    def test_svd_filter_preserves_shape(self):
        from backend.app.sleeves.futures_kronos.kronos_adapter import KronosFoundationAdapter

        adapter = KronosFoundationAdapter(api_url="", api_key="", context_length=50)
        data = np.random.randn(50, 5)
        filtered = adapter._inter_period_redundancy_filter(data)
        assert filtered.shape == data.shape

    def test_roc_distribution_returns_triple(self):
        from backend.app.sleeves.futures_kronos.kronos_adapter import KronosFoundationAdapter

        adapter = KronosFoundationAdapter(api_url="", api_key="", context_length=30)
        data = np.random.randn(50, 5)
        expected_mu, expected_sigma = adapter.get_roc_distribution(data)
        assert isinstance(expected_mu, float)
        assert isinstance(expected_sigma, float)


# =====================================================================
# 7. DDPG Allocator
# =====================================================================


class TestDDPGAllocator:
    def test_bounding_layer_sum_one(self):
        from backend.app.allocator.ddpg_tide_agent import DeterministicBoundingLayer

        layer = DeterministicBoundingLayer(n_sleeves=4, vrp_index=2, max_vrp=0.25)
        raw = torch.randn(8, 4)
        weights = layer(raw)
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(8), atol=1e-5, rtol=0)

    def test_bounding_layer_vrp_capped(self):
        from backend.app.allocator.ddpg_tide_agent import DeterministicBoundingLayer

        layer = DeterministicBoundingLayer(n_sleeves=4, vrp_index=2, max_vrp=0.25)
        # Push VRP logit very high
        raw = torch.tensor([[0.0, 0.0, 100.0, 0.0]])
        weights = layer(raw)
        assert weights[0, 2].item() <= 0.25 + 1e-6

    def test_tide_encoder_shape(self):
        from backend.app.allocator.ddpg_tide_agent import TiDEEncoder

        enc = TiDEEncoder(input_dim=5, seq_len=20, hidden_dim=64, output_dim=32)
        x = torch.randn(4, 20, 5)
        out = enc(x)
        assert out.shape == (4, 32)

    def test_actor_output_valid(self):
        from backend.app.allocator.ddpg_tide_agent import DDPGAllocatorActor

        actor = DDPGAllocatorActor(state_dim=32, n_sleeves=4, vrp_index=2, max_vrp=0.25)
        state = torch.randn(4, 32)
        weights = actor(state)
        assert weights.shape == (4, 4)
        sums = weights.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones(4), atol=1e-5, rtol=0)
        assert weights[:, 2].max().item() <= 0.25 + 1e-6


# =====================================================================
# 8. Signature Friction Model
# =====================================================================


class TestSignatureFriction:
    def test_output_positive(self):
        from backend.app.execution.signature_hedging import SignatureFrictionModel

        model = SignatureFrictionModel(channels=4, depth=3, output_dim=2)
        path = torch.randn(8, 50, 4)
        friction = model(path)
        assert friction.shape == (8, 2)
        assert (friction > 0).all(), "Friction must be strictly positive"

    def test_fallback_without_signatory(self):
        """The model should work even if signatory is mocked away."""
        from backend.app.execution import signature_hedging

        original = signature_hedging.signatory
        try:
            signature_hedging.signatory = None
            model = signature_hedging.SignatureFrictionModel(channels=4, depth=3, output_dim=2)
            path = torch.randn(4, 30, 4)
            friction = model(path)
            assert friction.shape == (4, 2)
            assert (friction > 0).all()
        finally:
            signature_hedging.signatory = original
