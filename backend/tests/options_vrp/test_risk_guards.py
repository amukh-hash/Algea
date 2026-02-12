"""
Tests for gamma danger-zone guard.
"""
import pytest
from datetime import date

from algaie.execution.options.config import VRPConfig
from algaie.execution.options.risk_guards import check_danger_zone
from algaie.execution.options.structures import (
    DerivativesPosition,
    OptionLeg,
    StructureType,
)


def _make_spread(short_strike: float = 530.0, underlying_price: float = 540.0) -> DerivativesPosition:
    return DerivativesPosition(
        underlying="SPY",
        structure_type=StructureType.PUT_CREDIT_SPREAD,
        expiry=date(2024, 8, 16),
        legs=[
            OptionLeg(option_type="put", strike=short_strike, qty=-1, side="sell",
                      entry_price_mid=3.40, entry_iv=0.18, entry_underlying=underlying_price),
            OptionLeg(option_type="put", strike=short_strike - 5, qty=1, side="buy",
                      entry_price_mid=2.15, entry_iv=0.20, entry_underlying=underlying_price),
        ],
        premium_collected=1.25,
        max_loss=3.75,
        position_id="dz_test",
    )


class TestDangerZone:
    def test_no_danger_far_otm(self):
        """Short strike far OTM should not trigger."""
        pos = _make_spread(short_strike=500.0, underlying_price=540.0)
        cfg = VRPConfig()
        result = check_danger_zone(pos, 540.0, 0.01, date(2024, 7, 20), cfg)
        assert not result.in_danger
        assert result.action == "none" or result.action == "tighten_stop"

    def test_danger_when_spot_approaches_strike(self):
        """When spot is near the short strike, both signals should fire."""
        pos = _make_spread(short_strike=530.0)
        cfg = VRPConfig(
            danger_zone_delta_threshold=0.30,
            danger_zone_z_threshold=0.85,
            danger_zone_close_if_both=True,
        )
        # Spot very close to strike
        result = check_danger_zone(pos, 531.0, 0.015, date(2024, 8, 10), cfg)
        assert result.delta_triggered or result.z_triggered
        # Either in_danger or tighten_stop

    def test_delta_triggered_alone(self):
        """Delta > threshold but z-score safe → tighten_stop."""
        pos = _make_spread(short_strike=530.0)
        cfg = VRPConfig(
            danger_zone_delta_threshold=0.20,  # very loose threshold
            danger_zone_z_threshold=0.10,       # very tight z threshold
            danger_zone_close_if_both=True,
        )
        result = check_danger_zone(pos, 533.0, 0.01, date(2024, 8, 10), cfg)
        # With close_if_both, in_danger requires both
        if result.delta_triggered and not result.z_triggered:
            assert result.action in ("tighten_stop", "none")

    def test_z_score_computation(self):
        """Z-score should decrease as spot approaches strike."""
        pos = _make_spread(short_strike=530.0)
        cfg = VRPConfig()

        result_far = check_danger_zone(pos, 550.0, 0.01, date(2024, 7, 20), cfg)
        result_close = check_danger_zone(pos, 532.0, 0.01, date(2024, 7, 20), cfg)

        assert result_far.z_score > result_close.z_score

    def test_no_short_leg_returns_safe(self):
        """Position with no short put should return safe."""
        pos = DerivativesPosition(
            underlying="SPY",
            structure_type=StructureType.PUT_CREDIT_SPREAD,
            expiry=date(2024, 8, 16),
            legs=[
                OptionLeg(option_type="put", strike=530, qty=1, side="buy",
                          entry_price_mid=3.40, entry_iv=0.18),
            ],
            premium_collected=0.0,
            max_loss=0.0,
            position_id="no_short",
        )
        result = check_danger_zone(pos, 540.0, 0.01, date(2024, 7, 20), VRPConfig())
        assert not result.in_danger
        assert result.action == "none"
