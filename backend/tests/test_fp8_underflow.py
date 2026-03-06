"""Test FP8 underflow prevention via BFloat16 autocast.

Validates that near-zero tensors (1e-7) through SMoEGateNet routing
produce non-zero gradients under the bfloat16 context, confirming
the FP8 underflow mitigation on Ada Lovelace GPUs.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from algaie.models.mera_scorer import SMoEGateNet, MERAEquityScorer


class TestFP8UnderflowPrevention:
    def test_near_zero_gradients_nonzero(self):
        """Near-zero (1e-7) inputs through SMoE gate produce non-zero gradients."""
        gate = SMoEGateNet(
            input_dim=32, n_experts=8, top_k=2,
            expert_dim=16, output_dim=8, market_dim=16,
        )
        gate.train()

        # Near-zero inputs that would underflow under FP8
        x = torch.ones(4, 32) * 1e-7
        x.requires_grad = True
        market = torch.ones(4, 16) * 1e-7
        market.requires_grad = True

        output, gate_probs = gate(x, market)
        loss = output.sum()
        loss.backward()

        # Gradients must not underflow to zero
        assert x.grad is not None, "No gradient for input"
        assert not torch.all(x.grad == 0), (
            "All gradients are zero — FP8 underflow likely occurring. "
            "Verify bfloat16 autocast is active in SMoEGateNet."
        )

    def test_gate_probs_sum_to_one(self):
        """Gate probabilities sum to ~1.0 even with tiny inputs."""
        gate = SMoEGateNet(
            input_dim=32, n_experts=8, top_k=2,
            expert_dim=16, output_dim=8, market_dim=16,
        )
        gate.eval()

        x = torch.ones(4, 32) * 1e-7
        market = torch.ones(4, 16) * 1e-7

        with torch.no_grad():
            _, gate_probs = gate(x, market)

        # Probabilities must sum to ~1.0
        sums = gate_probs.sum(dim=-1)
        torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-3, rtol=1e-3)

    def test_no_nan_in_routing(self):
        """Routing logits produce no NaN/Inf even with extreme inputs."""
        gate = SMoEGateNet(
            input_dim=32, n_experts=8, top_k=2,
            expert_dim=16, output_dim=8, market_dim=16,
        )
        gate.eval()

        # Very small and very large inputs
        for scale in [1e-7, 1e-3, 1.0, 1e3]:
            x = torch.ones(4, 32) * scale
            market = torch.ones(4, 16) * scale

            with torch.no_grad():
                output, gate_probs = gate(x, market)

            assert not torch.isnan(output).any(), f"NaN in output at scale {scale}"
            assert not torch.isinf(output).any(), f"Inf in output at scale {scale}"
            assert not torch.isnan(gate_probs).any(), f"NaN in gate_probs at scale {scale}"

    def test_mera_scorer_end_to_end(self):
        """Full MERAEquityScorer forward pass with near-zero inputs."""
        scorer = MERAEquityScorer(
            realtime_dim=32, historical_dim=64,
            market_dim=16, n_experts=4, top_k=2,
        )
        scorer.train()

        rt = torch.ones(4, 32) * 1e-7
        hist = torch.ones(4, 64) * 1e-7
        mkt = torch.ones(4, 16) * 1e-7

        scores, routing_loss = scorer(rt, hist, mkt)

        assert scores.shape == (4, 1)
        assert not torch.isnan(scores).any()
        assert routing_loss.item() >= 0
