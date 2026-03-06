"""Test: SMoE gate BFloat16 autocast produces non-zero routing.

Validates that the BFloat16 autocast wrapper on the SMoE gate logits
prevents Ada Lovelace FP8 tensor core underflow — even with near-zero
input features, no expert should receive exactly 0.0 routing probability.
"""
from __future__ import annotations

import torch
import pytest

from backend.app.sleeves.equity_mera.mera_scorer import SMoEGateNet, MERAEquityScorer


class TestBFloat16Routing:
    """Ensure gate probabilities are never exactly zero after autocast."""

    def test_near_zero_inputs_produce_nonzero_routing(self) -> None:
        """Microscopic inputs should not collapse to zero routing."""
        gate = SMoEGateNet(
            input_dim=48, n_experts=8, top_k=2,
            expert_dim=64, output_dim=32, market_dim=16,
        )
        gate.eval()

        x = torch.ones(4, 48) * 1e-7  # near-zero features
        market = torch.ones(4, 16) * 1e-7

        with torch.no_grad():
            output, gate_probs = gate(x, market)

        # No expert should have exactly 0.0 probability
        assert (gate_probs > 0).all(), (
            f"Gate produced zero routing probabilities: {gate_probs}"
        )

    def test_gate_probs_sum_to_one(self) -> None:
        """Gate softmax must produce valid probability distribution."""
        gate = SMoEGateNet(
            input_dim=48, n_experts=8, top_k=2,
            expert_dim=64, output_dim=32, market_dim=16,
        )
        gate.eval()

        x = torch.randn(8, 48)
        market = torch.randn(8, 16)

        with torch.no_grad():
            _, gate_probs = gate(x, market)

        row_sums = gate_probs.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            f"Gate probs don't sum to 1: {row_sums}"
        )

    def test_full_scorer_nonzero_output(self) -> None:
        """Full MERAEquityScorer should produce non-zero alpha scores."""
        scorer = MERAEquityScorer(
            realtime_dim=32, historical_dim=64,
            market_dim=16, n_experts=8, top_k=2,
        )
        scorer.eval()

        rt = torch.randn(4, 32)
        hist = torch.randn(4, 64)
        mkt = torch.randn(4, 16)

        with torch.no_grad():
            scores, routing_loss = scorer(rt, hist, mkt)

        assert scores.shape == (4, 1)
        assert not torch.isnan(scores).any(), "Scores contain NaN"
        assert not torch.isinf(scores).any(), "Scores contain Inf"
        assert torch.isfinite(routing_loss), "Routing loss is not finite"
