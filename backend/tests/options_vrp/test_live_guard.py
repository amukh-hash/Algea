"""Tests for paper/live safety guard (Phase 6)."""
from __future__ import annotations

from datetime import date

import pytest

from algae.execution.options.config import VRPConfig
from backend.app.risk.live_guard import LiveGuard, LiveGuardDecision


class TestScenarioGuard:
    def test_blocks_when_loss_exceeds_hard_limit(self):
        cfg = VRPConfig(live_guard_hard_loss_limit=0.08)
        guard = LiveGuard(cfg)
        decision = guard.evaluate(
            as_of_date=date(2024, 6, 1),
            scenario_loss_pct=0.10,
            margin_utilization=0.50,
            forecast_health=0.90,
        )
        assert not decision.allow_new_trades
        assert any("scenario_loss_pct" in r for r in decision.reasons)

    def test_allows_when_loss_below_limit(self):
        cfg = VRPConfig(live_guard_hard_loss_limit=0.08)
        guard = LiveGuard(cfg)
        decision = guard.evaluate(
            as_of_date=date(2024, 6, 1),
            scenario_loss_pct=0.04,
            margin_utilization=0.50,
            forecast_health=0.90,
        )
        assert decision.allow_new_trades


class TestMarginGuard:
    def test_blocks_when_margin_high(self):
        cfg = VRPConfig(live_guard_margin_limit=0.90)
        guard = LiveGuard(cfg)
        decision = guard.evaluate(
            as_of_date=date(2024, 6, 1),
            scenario_loss_pct=0.03,
            margin_utilization=0.95,
            forecast_health=0.90,
        )
        assert not decision.allow_new_trades
        assert any("margin" in r for r in decision.reasons)


class TestHealthDropGuard:
    def test_sudden_health_drop_reduces_vrp(self):
        cfg = VRPConfig(
            live_guard_health_drop_threshold=0.20,
            live_guard_health_drop_reduce_pct=0.50,
        )
        guard = LiveGuard(cfg)
        # 4 days of health: 0.90, 0.85, 0.75, 0.55 → drop = 0.35
        for h in [0.90, 0.85, 0.75]:
            guard.evaluate(date(2024, 6, 1), 0.03, 0.50, h)
        decision = guard.evaluate(date(2024, 6, 4), 0.03, 0.50, 0.55)
        assert decision.reduce_vrp_pct == 0.50

    def test_stable_health_no_reduction(self):
        cfg = VRPConfig(live_guard_health_drop_threshold=0.20)
        guard = LiveGuard(cfg)
        for h in [0.90, 0.88, 0.87, 0.86]:
            decision = guard.evaluate(date(2024, 6, 1), 0.03, 0.50, h)
        assert decision.reduce_vrp_pct == 0.0


class TestRegimeGuard:
    def test_crash_risk_blocks(self):
        guard = LiveGuard()
        decision = guard.evaluate(
            as_of_date=date(2024, 6, 1),
            scenario_loss_pct=0.02,
            margin_utilization=0.30,
            forecast_health=0.90,
            regime="crash_risk",
        )
        assert not decision.allow_new_trades

    def test_normal_regime_allows(self):
        guard = LiveGuard()
        decision = guard.evaluate(
            as_of_date=date(2024, 6, 1),
            scenario_loss_pct=0.02,
            margin_utilization=0.30,
            forecast_health=0.90,
            regime="normal_carry",
        )
        assert decision.allow_new_trades


class TestDecisionAudit:
    def test_serializable(self):
        guard = LiveGuard()
        decision = guard.evaluate(date(2024, 6, 1), 0.10, 0.95, 0.50, "crash_risk")
        d = decision.to_dict()
        assert "allow_new_trades" in d
        assert "reasons" in d
        assert len(d["reasons"]) > 0
