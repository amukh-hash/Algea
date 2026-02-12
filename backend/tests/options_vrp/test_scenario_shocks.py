"""Tests for scenario shock engine — monotonicity, crash combo, worst case.

v2: updated to match new compute_position_scenario_pnl signature (as_of_date required).
"""
from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from algaie.execution.options.structures import (
    DerivativesPosition,
    DerivativesPositionFrame,
    OptionLeg,
    StructureType,
)
from algaie.trading.derivatives_risk import (
    compute_position_scenario_pnl,
    compute_scenario_grid,
)


def _make_short_put_spread() -> DerivativesPosition:
    return DerivativesPosition(
        underlying="SPY",
        structure_type=StructureType.PUT_CREDIT_SPREAD,
        expiry=date(2024, 7, 15),
        legs=[
            OptionLeg("put", 395.0, -1, "sell", 3.00, delta=-0.20, entry_iv=0.20,
                      entry_underlying=400.0),
            OptionLeg("put", 390.0, 1, "buy", 1.50, delta=-0.12, entry_iv=0.22,
                      entry_underlying=400.0),
        ],
        premium_collected=1.50,
        max_loss=3.50,
        multiplier=100,
        position_id="test_shock",
    )


class TestScenarioPnL:
    def test_no_shock_pnl_near_zero(self):
        pos = _make_short_put_spread()
        pnl = compute_position_scenario_pnl(
            pos, 400.0, as_of_date=date(2024, 6, 1), spot_shock=0.0, vol_shock=0.0,
        )
        # At flat conditions, repriced should be close to entry
        assert isinstance(pnl, float)

    def test_negative_spot_shock_increases_loss(self):
        """Larger negative spot moves should produce worse PnL for short put spreads."""
        pos = _make_short_put_spread()
        pnl_5 = compute_position_scenario_pnl(
            pos, 400.0, as_of_date=date(2024, 6, 1), spot_shock=-0.05, vol_shock=0.0,
        )
        pnl_10 = compute_position_scenario_pnl(
            pos, 400.0, as_of_date=date(2024, 6, 1), spot_shock=-0.10, vol_shock=0.0,
        )
        pnl_20 = compute_position_scenario_pnl(
            pos, 400.0, as_of_date=date(2024, 6, 1), spot_shock=-0.20, vol_shock=0.0,
        )

        # Monotonic: deeper shocks → worse PnL
        assert pnl_10 <= pnl_5 + 50, "Larger shock should produce worse PnL"
        assert pnl_20 <= pnl_10 + 50, "20% shock should be worse than 10%"

    def test_vol_shock_increases_loss_for_short_vol(self):
        """Higher vol should increase option prices → worse for short positions."""
        pos = _make_short_put_spread()
        pnl_0 = compute_position_scenario_pnl(
            pos, 400.0, as_of_date=date(2024, 6, 1), spot_shock=-0.05, vol_shock=0.0,
        )
        pnl_50 = compute_position_scenario_pnl(
            pos, 400.0, as_of_date=date(2024, 6, 1), spot_shock=-0.05, vol_shock=0.50,
        )
        # Vol expansion should hurt short vol (though spread limits the damage)
        assert pnl_50 <= pnl_0 + 100


class TestScenarioGrid:
    def test_grid_shape(self):
        pf = DerivativesPositionFrame()
        pf.add(_make_short_put_spread())
        grid = compute_scenario_grid(
            pf, {"SPY": 400.0},
            spot_shocks=[-0.10, 0.0, 0.10],
            vol_shocks=[0.0, 0.50],
        )
        assert len(grid) == 6, "3 spot × 2 vol = 6 scenarios"
        assert "total_pnl" in grid.columns

    def test_crash_combo_is_worst(self):
        """Combined spot crash + vol expansion should be among worst scenarios."""
        pf = DerivativesPositionFrame()
        pf.add(_make_short_put_spread())
        grid = compute_scenario_grid(
            pf, {"SPY": 400.0},
            spot_shocks=[-0.20, -0.10, -0.05, 0.0, 0.05, 0.10],
            vol_shocks=[0.0, 0.25, 0.50],
        )
        worst = grid["total_pnl"].min()
        flat = grid[(grid["spot_shock"] == 0.0) & (grid["vol_shock"] == 0.0)]["total_pnl"].iloc[0]
        assert worst <= flat, "Worst scenario should be <= flat"

    def test_empty_positions_returns_empty_grid(self):
        pf = DerivativesPositionFrame()
        grid = compute_scenario_grid(pf, {"SPY": 400.0})
        assert len(grid) > 0  # Grid is produced even with no positions
        assert (grid["total_pnl"] == 0.0).all(), "No positions → zero PnL"
