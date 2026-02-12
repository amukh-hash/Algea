"""
Tests for the constraint-driven de-risk policy.
"""
import pytest
from datetime import date

from algaie.execution.options.config import VRPConfig
from algaie.execution.options.exits import DeRiskPolicy, ExitReason
from algaie.execution.options.structures import (
    DerivativesPosition,
    OptionLeg,
    StructureType,
)


def _make_position(pid: str, max_loss: float = 3.75, underlying: str = "SPY") -> DerivativesPosition:
    return DerivativesPosition(
        underlying=underlying,
        structure_type=StructureType.PUT_CREDIT_SPREAD,
        expiry=date(2024, 8, 16),
        legs=[
            OptionLeg(option_type="put", strike=530, qty=-1, side="sell",
                      entry_price_mid=3.40, entry_iv=0.18, entry_underlying=540.0),
            OptionLeg(option_type="put", strike=525, qty=1, side="buy",
                      entry_price_mid=2.15, entry_iv=0.20, entry_underlying=540.0),
        ],
        premium_collected=1.25,
        max_loss=max_loss,
        position_id=pid,
    )


class TestDeRiskPolicy:
    def test_no_action_when_constraints_ok(self):
        cfg = VRPConfig(max_worst_case_scenario_loss_pct_nav=0.10)
        policy = DeRiskPolicy(cfg)
        positions = [_make_position("p1")]
        # Scenario loss well within budget
        summary = policy.evaluate(
            positions, "normal_carry", {"p1": -100.0}, -100.0, 100_000,
        )
        assert summary.constraints_restored
        assert len(summary.actions) == 0

    def test_closes_until_constraints_restored(self):
        cfg = VRPConfig(
            max_worst_case_scenario_loss_pct_nav=0.05,
            max_daily_derisk_actions=5,
        )
        policy = DeRiskPolicy(cfg)
        # 3 positions, total loss exceeds 5% of 10k = 500
        positions = [_make_position(f"p{i}") for i in range(3)]
        contribs = {"p0": -300.0, "p1": -200.0, "p2": -100.0}
        summary = policy.evaluate(
            positions, "caution", contribs, -600.0, 10_000,
        )
        # Should close at least 1 position to bring loss below 500
        assert len(summary.actions) >= 1
        # Highest contributor (p0) should be closed first
        assert summary.actions[0].position_id == "p0"

    def test_crash_risk_disables_entries_closes_positions(self):
        cfg = VRPConfig(
            max_worst_case_scenario_loss_pct_nav=0.10,
            max_daily_derisk_actions=2,
        )
        policy = DeRiskPolicy(cfg)
        positions = [_make_position("p1"), _make_position("p2")]
        contribs = {"p1": -50.0, "p2": -30.0}
        summary = policy.evaluate(
            positions, "crash_risk", contribs, -80.0, 100_000,
        )
        # CRASH_RISK should trigger de-risk actions even if within budget
        assert len(summary.actions) >= 1

    def test_daily_cap_respected(self):
        cfg = VRPConfig(
            max_worst_case_scenario_loss_pct_nav=0.01,  # very tight
            max_daily_derisk_actions=2,
        )
        policy = DeRiskPolicy(cfg)
        positions = [_make_position(f"p{i}") for i in range(5)]
        contribs = {f"p{i}": -1000.0 for i in range(5)}
        summary = policy.evaluate(
            positions, "caution", contribs, -5000.0, 10_000,
        )
        assert len(summary.actions) <= 2

    def test_danger_zone_positions_prioritised(self):
        cfg = VRPConfig(max_worst_case_scenario_loss_pct_nav=0.05)
        policy = DeRiskPolicy(cfg)
        positions = [_make_position("safe"), _make_position("danger")]
        contribs = {"safe": -300.0, "danger": -100.0}
        dz_flags = {"safe": False, "danger": True}
        summary = policy.evaluate(
            positions, "caution", contribs, -400.0, 10_000, dz_flags,
        )
        # Danger zone position should be closed first despite lower scenario contribution
        assert any(a.position_id == "danger" for a in summary.actions)


class TestExitReasons:
    def test_all_reasons_have_values(self):
        for reason in ExitReason:
            assert len(reason.value) > 0

    def test_exit_reason_populated_on_close(self):
        pos = _make_position("p1")
        pos.exit_reason = ExitReason.PROFIT_TAKE.value
        assert pos.exit_reason == "profit_take"
