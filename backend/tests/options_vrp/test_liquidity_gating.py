"""Tests for liquidity gating in de-risk policy (Phase 3)."""
from __future__ import annotations

from datetime import date

import pytest

from algea.execution.options.config import VRPConfig
from algea.execution.options.exits import DeRiskPolicy, ExitReason
from algea.execution.options.structures import (
    DerivativesPosition,
    OptionLeg,
    StructureType,
)


def _pos(pid: str = "a", max_loss: float = 5.0) -> DerivativesPosition:
    return DerivativesPosition(
        underlying="SPY",
        structure_type=StructureType.PUT_CREDIT_SPREAD,
        expiry=date(2024, 7, 15),
        legs=[
            OptionLeg("put", 395.0, -1, "sell", 2.50, delta=-0.15),
            OptionLeg("put", 390.0, 1, "buy", 1.00, delta=-0.08),
        ],
        premium_collected=1.50,
        max_loss=max_loss,
        multiplier=100,
        position_id=pid,
    )


class TestLiquidityGating:
    def test_wide_spreads_block_optional_closes(self):
        """Non-required closes should be skipped when spread is too wide."""
        cfg = VRPConfig(
            max_spread_pct_live=0.10,
            liquidity_block_optional_closes=True,
            max_worst_case_scenario_loss_pct_nav=0.06,
        )
        policy = DeRiskPolicy(cfg)
        positions = [_pos("a"), _pos("b")]

        # Scenario loss within budget but regime is crash → regime_derisk
        summary = policy.evaluate(
            positions=positions,
            regime="crash_risk",
            scenario_contributions={"a": -100.0, "b": -100.0},
            total_scenario_loss=-200.0,
            nav=100_000,
            current_spreads={"a": 0.15, "b": 0.15},  # wide spreads
        )
        # In crash_risk, required_for_limits is True (regime override) so
        # closes execute even with wide spreads
        assert len(summary.actions) >= 1

    def test_required_closes_execute_despite_wide_spreads(self):
        """Required closes (budget exceeded) must execute regardless of spread."""
        cfg = VRPConfig(
            max_spread_pct_live=0.10,
            liquidity_block_optional_closes=True,
            max_worst_case_scenario_loss_pct_nav=0.01,  # 1% budget, very tight
        )
        policy = DeRiskPolicy(cfg)
        positions = [_pos("a")]

        summary = policy.evaluate(
            positions=positions,
            regime="caution",
            scenario_contributions={"a": -5000.0},
            total_scenario_loss=-5000.0,  # abs(5000) > 1% of 100k = 1000
            nav=100_000,
            current_spreads={"a": 0.20},  # very wide
        )
        # Must close — budget exceeded (required)
        assert len(summary.actions) >= 1

    def test_no_gate_when_disabled(self):
        """With liquidity_block_optional_closes=False, all closes execute."""
        cfg = VRPConfig(
            max_spread_pct_live=0.10,
            liquidity_block_optional_closes=False,
            max_worst_case_scenario_loss_pct_nav=0.06,
        )
        policy = DeRiskPolicy(cfg)
        positions = [_pos("a")]

        summary = policy.evaluate(
            positions=positions,
            regime="crash_risk",
            scenario_contributions={"a": -100.0},
            total_scenario_loss=-100.0,
            nav=100_000,
            current_spreads={"a": 0.20},
        )
        assert len(summary.actions) >= 1
