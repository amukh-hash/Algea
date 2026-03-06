"""Tests for risk limit enforcement."""
from __future__ import annotations

from datetime import date

import pytest

from algae.execution.options.config import VRPConfig
from algae.execution.options.structures import (
    DerivativesPosition,
    DerivativesPositionFrame,
    OptionLeg,
    StructureType,
)
from algae.trading.derivatives_risk import (
    check_risk_limits,
    compute_capital_at_risk,
    compute_max_loss,
)


def _make_position(
    max_loss: float = 4.0,
    multiplier: int = 100,
    underlying: str = "SPY",
    position_id: str = "test1",
) -> DerivativesPosition:
    return DerivativesPosition(
        underlying=underlying,
        structure_type=StructureType.PUT_CREDIT_SPREAD,
        expiry=date(2024, 7, 15),
        legs=[
            OptionLeg("put", 395.0, -1, "sell", 2.50, delta=-0.15),
            OptionLeg("put", 390.0, 1, "buy", 1.00, delta=-0.08),
        ],
        premium_collected=1.50,
        max_loss=max_loss,
        multiplier=multiplier,
        delta=-0.07,
        vega=-5.0,
        gamma=-0.5,
        position_id=position_id,
    )


class TestMaxLoss:
    def test_max_loss_calculation(self):
        pos = _make_position(max_loss=4.0, multiplier=100)
        assert compute_max_loss(pos) == 400.0

    def test_capital_at_risk_sums_open(self):
        pf = DerivativesPositionFrame()
        pf.add(_make_position(max_loss=4.0, position_id="a"))
        pf.add(_make_position(max_loss=3.0, position_id="b"))
        assert compute_capital_at_risk(pf) == 700.0


class TestRiskLimits:
    def test_within_limits_passes(self):
        """Position within all limits should pass."""
        config = VRPConfig(
            max_risk_per_structure_pct_nav=0.02,
            max_total_vrp_risk_pct_nav=0.10,
        )
        pf = DerivativesPositionFrame()
        pf.add(_make_position(max_loss=4.0, position_id="a"))
        result = check_risk_limits(pf, nav=100_000, config=config)
        assert result.passed, f"Should pass: {result.violations}"

    def test_per_structure_violation(self):
        """Position exceeding per-structure limit should fail."""
        config = VRPConfig(max_risk_per_structure_pct_nav=0.002, budget_basis="risk")  # 0.2%
        pf = DerivativesPositionFrame()
        pf.add(_make_position(max_loss=4.0, position_id="a"))
        # max_loss = 400 / 100_000 = 0.4% > 0.2%
        result = check_risk_limits(pf, nav=100_000, config=config)
        assert not result.passed
        assert any("budget" in v.lower() or "max_loss" in v.lower() for v in result.violations)

    def test_aggregate_violation(self):
        """Total risk exceeding aggregate limit should fail."""
        config = VRPConfig(max_total_vrp_risk_pct_nav=0.005)  # 0.5%
        pf = DerivativesPositionFrame()
        pf.add(_make_position(max_loss=4.0, position_id="a"))
        pf.add(_make_position(max_loss=4.0, position_id="b"))
        # total = 800 / 100_000 = 0.8% > 0.5%
        result = check_risk_limits(pf, nav=100_000, config=config)
        assert not result.passed
        assert any("Total VRP" in v for v in result.violations)

    def test_position_count_violation(self):
        """Exceeded position count per underlying should fail."""
        config = VRPConfig(max_positions_per_underlying=1)
        pf = DerivativesPositionFrame()
        pf.add(_make_position(position_id="a", underlying="SPY"))
        pf.add(_make_position(position_id="b", underlying="SPY"))
        result = check_risk_limits(pf, nav=1_000_000, config=config)
        assert not result.passed
        assert any("positions" in v.lower() for v in result.violations)
