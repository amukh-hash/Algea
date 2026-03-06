"""
Tests for MetaAllocator stability — EMA smoothing, daily delta cap,
min hold days, crash override.
"""
from datetime import date

import pytest

from algae.data.options.vrp_features import VolRegime
from algae.execution.options.config import VRPConfig
from algae.trading.meta_allocator import (
    AllocationResult,
    AllocatorState,
    MetaAllocator,
    SleeveResult,
)


def _equity_sleeve() -> SleeveResult:
    return SleeveResult(name="equity", expected_return=0.08, realized_vol=0.16)


def _vrp_sleeve(expected_return: float = 0.06, es_95: float = 0.10) -> SleeveResult:
    return SleeveResult(
        name="vrp", expected_return=expected_return, realized_vol=0.08,
        max_drawdown=0.05, es_95=es_95,
    )


class TestSmoothingAndCaps:
    def test_ema_smoothing_applied(self):
        """Weight should be smoothed from prior day."""
        cfg = VRPConfig(w_smoothing_alpha=0.25, w_max_daily_delta=0.10, w_min_hold_days=0)
        state = AllocatorState(w_vrp_prev=0.10)
        alloc = MetaAllocator(cfg, state)
        result = alloc.combine(
            date(2024, 7, 15), [_equity_sleeve(), _vrp_sleeve()], nav=1_000_000,
        )
        # Should be smoothed from 0.10 toward optimal
        assert result.smoothing_applied or abs(result.w_vrp - 0.10) < 0.05

    def test_daily_delta_cap(self):
        """Weight change should be capped at w_max_daily_delta."""
        cfg = VRPConfig(
            w_smoothing_alpha=1.0,  # No EMA, raw optimal
            w_max_daily_delta=0.01,
            w_min_hold_days=0,
        )
        state = AllocatorState(w_vrp_prev=0.10)
        alloc = MetaAllocator(cfg, state)
        result = alloc.combine(
            date(2024, 7, 15), [_equity_sleeve(), _vrp_sleeve()], nav=1_000_000,
        )
        assert abs(result.w_vrp - 0.10) <= cfg.w_max_daily_delta + 1e-6

    def test_min_hold_days(self):
        """Direction reversal should be suppressed within min_hold_days."""
        cfg = VRPConfig(
            w_smoothing_alpha=1.0,
            w_max_daily_delta=0.10,
            w_min_hold_days=5,
        )
        # State: weight was INCREASING (history shows 0.05 → 0.10)
        state = AllocatorState(
            w_vrp_prev=0.10,
            last_change_date=date(2024, 7, 14),
            weight_history=[0.05, 0.10],
        )
        alloc = MetaAllocator(cfg, state)
        # Switch to CAUTION (dynamic cap wants to pull weight DOWN → reversal)
        result = alloc.combine(
            date(2024, 7, 15), [_equity_sleeve(), _vrp_sleeve()], nav=1_000_000,
            regime=VolRegime.CAUTION,
        )
        # 1 day since last change, direction reversal → should hold
        assert abs(result.w_vrp - 0.10) < 0.002

    def test_crash_override_immediate_zero(self):
        """CRASH_RISK must set w_vrp=0 immediately, ignoring smoothing."""
        cfg = VRPConfig(w_smoothing_alpha=0.25, w_min_hold_days=5)
        state = AllocatorState(w_vrp_prev=0.20, last_change_date=date(2024, 7, 10))
        alloc = MetaAllocator(cfg, state)
        result = alloc.combine(
            date(2024, 7, 15), [_equity_sleeve(), _vrp_sleeve()],
            nav=1_000_000, regime=VolRegime.CRASH_RISK,
        )
        assert result.w_vrp == 0.0
        assert result.w_equity == 1.0


class TestForecastES:
    def test_forecast_risk_reduces_weight(self):
        """Higher forecast ES should reduce VRP weight."""
        cfg = VRPConfig(w_smoothing_alpha=1.0, w_max_daily_delta=1.0, w_min_hold_days=0)
        # No forecast risk
        result_lo = MetaAllocator(cfg, AllocatorState()).combine(
            date(2024, 7, 15),
            [_equity_sleeve(), _vrp_sleeve(es_95=0.05)],
            nav=1_000_000,
        )
        # High forecast risk
        vrp_hi = _vrp_sleeve(es_95=0.05)
        vrp_hi.forecast_risk = {0.95: 0.50}  # very high forecast RV
        result_hi = MetaAllocator(cfg, AllocatorState()).combine(
            date(2024, 7, 15),
            [_equity_sleeve(), vrp_hi],
            nav=1_000_000,
        )
        assert result_hi.w_vrp <= result_lo.w_vrp


class TestEquityOnly:
    def test_equity_only_sleeves(self):
        """With no VRP sleeve, should default to 100% equity."""
        allocator = MetaAllocator()
        sleeves = [SleeveResult(name="equity", expected_return=0.08, realized_vol=0.16)]
        result = allocator.combine(date(2024, 6, 1), sleeves, nav=1_000_000)
        assert result.w_vrp == 0.0
        assert result.w_equity == 1.0
