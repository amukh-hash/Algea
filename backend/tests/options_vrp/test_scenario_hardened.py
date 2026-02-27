"""
Tests for hardened scenario engine — per-leg IV, conservative marks,
dynamic shocks, concentration, margin.
"""
import pytest
from datetime import date

import numpy as np
import pandas as pd

from algea.execution.options.config import VRPConfig
from algea.execution.options.structures import (
    DerivativesPosition,
    DerivativesPositionFrame,
    OptionLeg,
    StructureType,
)
from algea.trading.derivatives_risk import (
    build_dynamic_shock_grid,
    check_expiry_concentration,
    check_strike_spacing,
    compute_budget_basis,
    compute_position_scenario_pnl,
    compute_risk_scaler,
    compute_scenario_with_contributors,
    estimate_margin,
)


def _make_pos(
    pid: str = "s1",
    short_iv: float = 0.20,
    long_iv: float = 0.22,
    short_strike: float = 530.0,
    long_strike: float = 525.0,
    underlying_price: float = 540.0,
    expiry: date = date(2024, 8, 16),
) -> DerivativesPosition:
    return DerivativesPosition(
        underlying="SPY",
        structure_type=StructureType.PUT_CREDIT_SPREAD,
        expiry=expiry,
        legs=[
            OptionLeg(option_type="put", strike=short_strike, qty=-1, side="sell",
                      entry_price_mid=3.40, entry_iv=short_iv,
                      entry_underlying=underlying_price, entry_mid=3.40),
            OptionLeg(option_type="put", strike=long_strike, qty=1, side="buy",
                      entry_price_mid=2.15, entry_iv=long_iv,
                      entry_underlying=underlying_price, entry_mid=2.15),
        ],
        premium_collected=1.25,
        max_loss=3.75,
        position_id=pid,
    )


class TestScenarioRepricing:
    def test_uses_entry_iv_not_placeholder(self):
        """Repricing must use the leg's entry_iv, not hardcoded 0.20."""
        pos = _make_pos(short_iv=0.35, long_iv=0.38)
        pnl_35 = compute_position_scenario_pnl(
            pos, 540.0, date(2024, 7, 20), spot_shock=0.0, vol_shock=0.0,
        )
        pos2 = _make_pos(short_iv=0.15, long_iv=0.17)
        pnl_15 = compute_position_scenario_pnl(
            pos2, 540.0, date(2024, 7, 20), spot_shock=0.0, vol_shock=0.0,
        )
        # Different IVs should produce different PnLs
        assert abs(pnl_35 - pnl_15) > 0.01

    def test_conservative_marks_increase_losses(self):
        """Liquidity widening should make scenario losses worse."""
        pos = _make_pos()
        pnl_normal = compute_position_scenario_pnl(
            pos, 540.0, date(2024, 7, 20), spot_shock=-0.10, vol_shock=0.50,
            liquidity_widen_factor=1.0,
        )
        pnl_wide = compute_position_scenario_pnl(
            pos, 540.0, date(2024, 7, 20), spot_shock=-0.10, vol_shock=0.50,
            liquidity_widen_factor=3.0,
        )
        # Wider liquidity should cause worse (more negative) PnL
        assert pnl_wide <= pnl_normal

    def test_scenario_contributors_sorted(self):
        """Per-position contributions should be computed correctly."""
        frame = DerivativesPositionFrame()
        frame.add(_make_pos("p1"))
        frame.add(_make_pos("p2", short_strike=520, long_strike=515))
        total, contribs = compute_scenario_with_contributors(
            frame, {"SPY": 540.0}, date(2024, 7, 20),
        )
        assert len(contribs) == 2
        assert total <= 0  # scenario losses should be negative


class TestDynamicShocks:
    def test_fixed_shocks_always_included(self):
        spots, vols = build_dynamic_shock_grid()
        assert -0.10 in spots
        assert -0.05 in spots
        assert 0.25 in vols

    def test_high_rv_increases_shocks(self):
        """High predicted RV should add larger shocks."""
        cfg = VRPConfig(use_dynamic_shocks=True, dynamic_shock_k=2.5)
        spots, _ = build_dynamic_shock_grid(
            rv10_pred_p95=0.60, rv10_pred_p99=0.80, config=cfg,
        )
        # Should have added shocks beyond the fixed -20%
        min_shock = min(spots)
        assert min_shock < -0.20

    def test_dynamic_shocks_disabled(self):
        cfg = VRPConfig(use_dynamic_shocks=False)
        spots, vols = build_dynamic_shock_grid(
            rv10_pred_p95=0.80, config=cfg,
        )
        # Only fixed shocks
        assert min(spots) >= -0.20


class TestConcentration:
    def test_expiry_concentration_limit(self):
        cfg = VRPConfig(max_loss_pct_single_expiry=0.50)
        frame = DerivativesPositionFrame()
        frame.add(_make_pos("p1", expiry=date(2024, 8, 16)))
        frame.add(_make_pos("p2", expiry=date(2024, 8, 16)))
        frame.add(_make_pos("p3", expiry=date(2024, 9, 20)))
        result = check_expiry_concentration(frame, cfg)
        # 2 of 3 positions on same expiry = 66.7% > 50%
        assert not result.passed

    def test_strike_spacing_enforced(self):
        cfg = VRPConfig(min_short_strike_spacing_pct=0.02)
        frame = DerivativesPositionFrame()
        # Two positions with short strikes very close (530 and 531 on SPY ~540)
        frame.add(_make_pos("p1", short_strike=530))
        frame.add(_make_pos("p2", short_strike=531))
        result = check_strike_spacing(frame, {"SPY": 540.0}, cfg)
        # Spacing = 1/540 = 0.19% < 2%
        assert not result.passed


class TestMarginAndBudget:
    def test_margin_exceeds_max_loss(self):
        pos = _make_pos()
        cfg = VRPConfig(margin_buffer_multiplier=1.4)
        margin = estimate_margin(pos, cfg)
        max_loss = pos.max_loss * pos.multiplier
        assert margin > max_loss

    def test_budget_basis_margin(self):
        pos = _make_pos()
        cfg = VRPConfig(budget_basis="margin")
        budget = compute_budget_basis(pos, cfg)
        max_loss = pos.max_loss * pos.multiplier
        assert budget >= max_loss

    def test_budget_basis_risk(self):
        pos = _make_pos()
        cfg = VRPConfig(budget_basis="risk")
        budget = compute_budget_basis(pos, cfg)
        max_loss = pos.max_loss * pos.multiplier
        assert budget == max_loss


class TestRiskScaler:
    def test_scaler_bounded(self):
        pnl = pd.Series(np.random.randn(50) * 100)
        scaler = compute_risk_scaler(pnl, target_vol=0.08)
        assert 0.25 <= scaler.final_scaler <= 2.0

    def test_forecast_reduces_scaler(self):
        """High forecast RV should reduce scaler."""
        pnl = pd.Series(np.random.randn(50) * 10)
        scaler_no_fc = compute_risk_scaler(pnl, target_vol=0.08)
        scaler_hi_fc = compute_risk_scaler(pnl, target_vol=0.08, forecast_rv_p50=0.50)
        assert scaler_hi_fc.final_scaler <= scaler_no_fc.final_scaler
