"""Tests for MetaAllocator — grid optimisation, regime override, stability.

v2: updated to match new SleeveResult API and AllocationResult fields.
"""
from __future__ import annotations

from dataclasses import asdict
from datetime import date

import numpy as np
import pandas as pd
import pytest

from algaie.data.options.vrp_features import VolRegime
from algaie.execution.options.config import VRPConfig
from algaie.trading.meta_allocator import AllocationResult, MetaAllocator, SleeveResult


def _make_sleeves(
    eq_ret: float = 0.08,
    eq_vol: float = 0.16,
    vrp_ret: float = 0.06,
    vrp_vol: float = 0.08,
) -> list[SleeveResult]:
    return [
        SleeveResult(
            name="equity",
            expected_return=eq_ret,
            realized_vol=eq_vol,
        ),
        SleeveResult(
            name="vrp",
            expected_return=vrp_ret,
            realized_vol=vrp_vol,
            es_95=0.10,
        ),
    ]


class TestMetaAllocator:
    def test_w_vrp_within_bounds(self):
        """VRP allocation should be ≤ w_max_vrp."""
        config = VRPConfig(w_max_vrp=0.25)
        allocator = MetaAllocator(config)
        result = allocator.combine(
            date(2024, 6, 1), _make_sleeves(), nav=1_000_000,
        )
        assert 0.0 <= result.w_vrp <= 0.25
        assert abs(result.w_equity + result.w_vrp - 1.0) < 1e-8

    def test_crash_risk_forces_zero_vrp(self):
        """CRASH_RISK regime should force w_vrp = 0."""
        allocator = MetaAllocator()
        result = allocator.combine(
            date(2024, 6, 1), _make_sleeves(), nav=1_000_000,
            regime=VolRegime.CRASH_RISK,
        )
        assert result.w_vrp == 0.0
        assert result.w_equity == 1.0

    def test_normal_regime_may_allocate_vrp(self):
        """In normal regime, allocator may choose non-zero VRP weight."""
        allocator = MetaAllocator()
        result = allocator.combine(
            date(2024, 6, 1), _make_sleeves(), nav=1_000_000,
            regime=VolRegime.NORMAL_CARRY,
        )
        # w_vrp could be 0 or positive — just verify it's valid
        assert result.w_vrp >= 0.0
        assert result.tail_adjusted_sharpe is not None

    def test_allocation_stability(self):
        """Small perturbations in inputs should not cause large allocation jumps."""
        config = VRPConfig(w_smoothing_alpha=1.0, w_max_daily_delta=1.0, w_min_hold_days=0)
        allocator = MetaAllocator(config)
        base = allocator.combine(
            date(2024, 6, 1), _make_sleeves(eq_vol=0.16), nav=1_000_000,
        )
        allocator2 = MetaAllocator(config)
        perturbed = allocator2.combine(
            date(2024, 6, 1), _make_sleeves(eq_vol=0.161), nav=1_000_000,
        )
        assert abs(base.w_vrp - perturbed.w_vrp) <= 0.05, \
            "Small equity vol change should not cause large allocation change"

    def test_equity_only_sleeves(self):
        """With no VRP sleeve, should default to 100% equity."""
        allocator = MetaAllocator()
        sleeves = [SleeveResult(name="equity", expected_return=0.08, realized_vol=0.16)]
        result = allocator.combine(
            date(2024, 6, 1), sleeves, nav=1_000_000,
        )
        assert result.w_vrp == 0.0
        assert result.w_equity == 1.0

    def test_allocation_result_serializable(self):
        allocator = MetaAllocator()
        result = allocator.combine(
            date(2024, 6, 1), _make_sleeves(), nav=1_000_000,
        )
        d = asdict(result)
        assert "w_equity" in d
        assert "w_vrp" in d
        assert "regime" in d
