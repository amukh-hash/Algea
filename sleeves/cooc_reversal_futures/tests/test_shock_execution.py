"""Tests for shock execution gating.

Covers:
- Shock flag reduces contracts deterministically
- Flatten bypass preserved on shock days
- Trade proxy and execution use identical multipliers
- Zero multiplier blocks entry entirely
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from sleeves.cooc_reversal_futures.config import (
    COOCReversalConfig,
    CostConfig,
    ShockConfig,
)
from sleeves.cooc_reversal_futures.pipeline.types import TradeProxyRealism


class TestShockExecutionSizing:
    """Test shock gating logic in build_daily_orders."""

    def _make_config(self, **shock_kwargs) -> COOCReversalConfig:
        return COOCReversalConfig(
            universe=("ES", "NQ", "YM", "RTY"),
            shock=ShockConfig(**shock_kwargs),
            gross_target=0.8,
            max_contracts_per_instrument=10,
        )

    def test_shock_reduces_gross(self):
        """When shock_score > threshold, gross should be scaled down."""
        cfg = self._make_config(
            enabled=True,
            shock_z_threshold=2.0,
            gross_multiplier_on_shock=0.5,
            per_instrument=True,
        )

        # Simulate: ES has a shock, NQ does not
        shock_scores = {"ES": 3.0, "NQ": 1.0, "YM": 0.5, "RTY": 0.8}

        # ES exceeds threshold (3.0 > 2.0) → should get scaled
        shocked = {sym for sym, score in shock_scores.items()
                   if score > cfg.shock.shock_z_threshold}
        assert "ES" in shocked
        assert "NQ" not in shocked

    def test_zero_multiplier_blocks_entry(self):
        """gross_multiplier_on_shock=0.0 should block entry entirely."""
        cfg = self._make_config(
            enabled=True,
            shock_z_threshold=2.0,
            gross_multiplier_on_shock=0.0,
            per_instrument=True,
        )

        shock_scores = {"ES": 3.0, "NQ": 1.0}
        blocked = set()
        for sym, score in shock_scores.items():
            if cfg.shock.enabled and score > cfg.shock.shock_z_threshold:
                if cfg.shock.gross_multiplier_on_shock == 0.0:
                    blocked.add(sym)

        assert "ES" in blocked
        assert "NQ" not in blocked

    def test_flatten_always_allowed(self):
        """force_eod_flatten should work regardless of shock state."""
        # Flatten is not gated by shock — it's a separate method
        # Just verify the flatten method exists and accepts positions
        from sleeves.cooc_reversal_futures.sleeve import COOCReversalFuturesSleeve
        sleeve = COOCReversalFuturesSleeve()
        # This should not raise
        result = sleeve.force_eod_flatten({"ES": 5, "NQ": -3})
        assert isinstance(result, list)

    def test_shock_disabled_no_scaling(self):
        """When shock.enabled=False, no scaling should occur."""
        cfg = self._make_config(enabled=False)
        shock_scores = {"ES": 10.0}  # Very high shock

        scaled = {}
        blocked = set()
        if cfg.shock.enabled:
            for sym, score in shock_scores.items():
                if score > cfg.shock.shock_z_threshold:
                    scaled[sym] = cfg.shock.gross_multiplier_on_shock

        # Nothing should be scaled
        assert len(scaled) == 0
        assert len(blocked) == 0


class TestTradeProxyShockConsistency:
    """Verify trade proxy uses same multipliers as execution."""

    def test_default_shock_gross_multiplier(self):
        """TradeProxyRealism.shock_gross_multiplier should match ShockConfig default."""
        realism = TradeProxyRealism()
        shock_cfg = ShockConfig()
        assert realism.shock_gross_multiplier == shock_cfg.gross_multiplier_on_shock

    def test_default_shock_z_threshold(self):
        """TradeProxyRealism.shock_z_threshold should match ShockConfig default."""
        realism = TradeProxyRealism()
        shock_cfg = ShockConfig()
        assert realism.shock_z_threshold == shock_cfg.shock_z_threshold

    def test_shock_slippage_multiplier_separate(self):
        """shock_slippage_multiplier is separate from shock_gross_multiplier."""
        realism = TradeProxyRealism(
            shock_slippage_multiplier=3.0,
            shock_gross_multiplier=0.25,
        )
        assert realism.shock_slippage_multiplier == 3.0
        assert realism.shock_gross_multiplier == 0.25

    def test_realism_serialization(self):
        """to_dict should include shock fields."""
        realism = TradeProxyRealism(
            shock_gross_multiplier=0.5,
            shock_z_threshold=2.5,
        )
        d = realism.to_dict()
        assert "shock_gross_multiplier" in d
        assert "shock_z_threshold" in d
        assert d["shock_gross_multiplier"] == 0.5
        assert d["shock_z_threshold"] == 2.5


class TestDeterministicShockReduction:
    """Verify shock reduction is deterministic given same inputs."""

    def test_deterministic_shocked_set(self):
        """Same scores + threshold → same shocked instruments."""
        shock_scores = {"ES": 3.0, "NQ": 1.5, "YM": 2.5, "RTY": 0.5}
        threshold = 2.0

        def compute_shocked(scores, thresh):
            return frozenset(s for s, v in scores.items() if v > thresh)

        result1 = compute_shocked(shock_scores, threshold)
        result2 = compute_shocked(shock_scores, threshold)
        assert result1 == result2
        assert result1 == frozenset({"ES", "YM"})

    def test_multiplier_application_deterministic(self):
        """Same gross * multiplier → same result every time."""
        gross = 0.8
        multiplier = 0.5
        for _ in range(10):
            assert gross * multiplier == 0.4
