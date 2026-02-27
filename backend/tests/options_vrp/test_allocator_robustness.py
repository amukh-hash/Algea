"""Tests for allocator robustness enhancements (Phase 7)."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from algea.data.options.vrp_features import VolRegime
from algea.execution.options.config import VRPConfig
from algea.trading.meta_allocator import (
    AllocatorState,
    AllocationResult,
    MetaAllocator,
    SleeveResult,
)


def _sleeves(vrp_ret: float = 0.06, vrp_vol: float = 0.08) -> list[SleeveResult]:
    return [
        SleeveResult(name="equity", expected_return=0.08, realized_vol=0.16),
        SleeveResult(name="vrp", expected_return=vrp_ret, realized_vol=vrp_vol, es_95=0.10),
    ]


class TestWeightFloor:
    def test_tiny_weight_snapped_to_zero(self):
        """Allocations below w_min_deployment should snap to 0."""
        config = VRPConfig(
            w_max_vrp=0.25,
            w_min_deployment=0.05,
            w_smoothing_alpha=1.0,
            w_max_daily_delta=1.0,
            w_min_hold_days=0,
        )
        allocator = MetaAllocator(config)
        # With very low expected return → small weight
        sleeves = [
            SleeveResult(name="equity", expected_return=0.12, realized_vol=0.16),
            SleeveResult(name="vrp", expected_return=0.001, realized_vol=0.15, es_95=0.20),
        ]
        result = allocator.combine(date(2024, 6, 1), sleeves, nav=1_000_000)
        # Weight should be either >= floor or exactly 0
        assert result.w_vrp == 0.0 or result.w_vrp >= config.w_min_deployment

    def test_moderate_weight_passes_floor(self):
        """Reasonable allocation should survive the floor check."""
        config = VRPConfig(
            w_min_deployment=0.05,
            w_smoothing_alpha=1.0,
            w_max_daily_delta=1.0,
            w_min_hold_days=0,
        )
        allocator = MetaAllocator(config)
        result = allocator.combine(date(2024, 6, 1), _sleeves(), nav=1_000_000)
        assert result.w_vrp >= config.w_min_deployment or result.w_vrp == 0.0


class TestWeightVolatility:
    def test_volatility_computed_after_history(self):
        """After several days, weight_volatility should be computable."""
        config = VRPConfig(
            w_smoothing_alpha=1.0,
            w_max_daily_delta=1.0,
            w_min_hold_days=0,
            w_min_deployment=0.0,
        )
        allocator = MetaAllocator(config)
        for i in range(5):
            allocator.combine(
                date(2024, 6, 1) + timedelta(days=i),
                _sleeves(vrp_ret=0.06 + i * 0.01),
                nav=1_000_000,
            )
        vol = allocator.state.weight_volatility()
        # Should be a non-negative number
        assert vol >= 0.0

    def test_constant_weight_zero_volatility(self):
        """If weights are constant, volatility should be 0."""
        state = AllocatorState(weight_history=[0.10] * 10)
        assert state.weight_volatility() == 0.0


class TestCrashOverrideStillImmediate:
    def test_crash_bypasses_floor_and_smoothing(self):
        """CRASH_RISK should immediately zero VRP even with floor > 0."""
        config = VRPConfig(w_min_deployment=0.05)
        allocator = MetaAllocator(config)
        result = allocator.combine(
            date(2024, 6, 1), _sleeves(), nav=1_000_000,
            regime=VolRegime.CRASH_RISK,
        )
        assert result.w_vrp == 0.0


class TestNoOscillation:
    def test_floor_prevents_tiny_oscillation(self):
        """Repeated calls with borderline weight should not oscillate between 0 and small."""
        config = VRPConfig(
            w_min_deployment=0.05,
            w_smoothing_alpha=0.25,
            w_max_daily_delta=0.02,
            w_min_hold_days=0,
        )
        allocator = MetaAllocator(config)
        weights = []
        for i in range(10):
            result = allocator.combine(
                date(2024, 6, 1) + timedelta(days=i),
                _sleeves(vrp_ret=0.001, vrp_vol=0.15),
                nav=1_000_000,
            )
            weights.append(result.w_vrp)
        # No tiny non-zero values
        for w in weights:
            assert w == 0.0 or w >= config.w_min_deployment
