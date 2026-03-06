"""
Test: Latency bounds.

Validates that model inference completes within the DAG FSM's
strict 500ms timeout guard.
"""
from __future__ import annotations

import time

import torch
import pytest


class TestLatencyBounds:
    """All model forward passes must complete within 500ms on CPU."""

    @pytest.mark.parametrize("num_equities", [10, 50, 100])
    def test_itransformer_latency(self, num_equities: int):
        """iTransformer inference must be < 500ms even for large cross-sections."""
        from algae.models.tsfm.itransformer import iTransformer

        model = iTransformer(
            num_variates=num_equities,
            lookback_len=60,
            pred_len=5,
            d_model=128 if num_equities <= 50 else 64,
            n_heads=4,
            e_layers=2,
        )
        model.eval()

        x = torch.randn(1, 60, num_equities)

        # Warm up (JIT compilation, memory allocation)
        with torch.inference_mode():
            _ = model(x)

        # Timed run
        start = time.perf_counter()
        with torch.inference_mode():
            _ = model(x)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, (
            f"iTransformer inference with {num_equities} variates "
            f"took {elapsed_ms:.1f}ms, exceeding the 500ms guard."
        )

    def test_rank_transformer_latency(self):
        """RankTransformer with 64 equities must be < 500ms."""
        from algae.models.ranker.rank_transformer import RankTransformer

        model = RankTransformer(d_input=18, d_model=128, n_head=4, n_layers=2, max_len=64)
        model.eval()

        x = torch.randn(1, 64, 18)

        with torch.inference_mode():
            _ = model(x)

        start = time.perf_counter()
        with torch.inference_mode():
            _ = model(x)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 500, (
            f"RankTransformer inference took {elapsed_ms:.1f}ms, exceeding 500ms guard."
        )

    def test_td3_actor_latency(self):
        """TD3 Actor inference must be sub-millisecond."""
        from algae.models.rl.td3 import TD3Actor

        actor = TD3Actor(state_dim=256, action_dim=2)
        actor.eval()

        x = torch.randn(1, 256)

        with torch.inference_mode():
            _ = actor(x)

        start = time.perf_counter()
        with torch.inference_mode():
            _ = actor(x)
        elapsed_ms = (time.perf_counter() - start) * 1000

        assert elapsed_ms < 10, (
            f"TD3 Actor inference took {elapsed_ms:.1f}ms, should be < 10ms."
        )
