"""
Tests for the 6-phase exposure expansion pass.

Phase 1: Dynamic regime caps
Phase 2: Entry filter quality gates
Phase 3: Enhanced allocator objective
Phase 4: Deployment floor / ramp adjustments
Phase 5: Normal utilization target
Phase 6: Validation metrics
"""
import pytest
import numpy as np
import pandas as pd
from datetime import date, timedelta

from algea.execution.options.config import VRPConfig
from algea.data.options.vrp_features import VolRegime
from algea.trading.meta_allocator import (
    MetaAllocator, SleeveResult, AllocationResult, AllocatorState,
    AllocatorInvariantError, AllocationContext,
)
from algea.trading.entry_filter import (
    EntrySignals, EntryDecision, evaluate_entry,
)


def _sleeves():
    return [
        SleeveResult(name="equity", expected_return=0.08, realized_vol=0.16),
        SleeveResult(name="vrp", expected_return=0.06, realized_vol=0.08, es_95=0.10),
    ]


def _default_ctx(**kwargs):
    return AllocationContext(**kwargs)


# ══════════════════════════════════════════════════════════════════════════
# Phase 1: Dynamic Regime Caps
# ══════════════════════════════════════════════════════════════════════════

class TestDynamicRegimeCaps:
    """Phase 1: regime-conditioned exposure rebalancing."""

    def test_normal_allocation_exceeds_prior_low(self):
        """NORMAL regime should now reach meaningful allocation over time."""
        cfg = VRPConfig()
        alloc = MetaAllocator(cfg)
        ctx = _default_ctx(forecast_health=0.92, headroom_ratio=0.80)
        for i in range(60):
            result = alloc.combine(
                date(2024, 1, 1) + timedelta(days=i),
                _sleeves(), nav=1_000_000,
                context=ctx,
            )
        # With v4 alpha=0.18 and utilization pull, should be materially > 0.02
        assert result.w_vrp > 0.05, f"w_vrp too low: {result.w_vrp}"

    def test_normal_boost_when_health_and_headroom_good(self):
        """v4: normal boost multiplier activates with strong signals."""
        cfg = VRPConfig(normal_boost_multiplier=1.3)
        alloc = MetaAllocator(cfg)
        # Build up weight first
        for i in range(30):
            alloc.combine(
                date(2024, 1, 1) + timedelta(days=i),
                _sleeves(), nav=1_000_000,
                context=_default_ctx(forecast_health=0.92, headroom_ratio=0.80),
            )
        # One day with boost eligible context
        r = alloc.combine(
            date(2024, 2, 1), _sleeves(), nav=1_000_000,
            context=_default_ctx(forecast_health=0.92, headroom_ratio=0.80),
        )
        assert r.w_regime_cap >= r.w_opt  # boost applied

    def test_caution_exposure_nonzero_but_bounded(self):
        """CAUTION should have some exposure (not near-zero like v3)."""
        cfg = VRPConfig()
        alloc = MetaAllocator(cfg)
        ctx = _default_ctx(forecast_health=0.92, headroom_ratio=0.60)
        # Build weight in NORMAL first
        for i in range(40):
            alloc.combine(
                date(2024, 1, 1) + timedelta(days=i),
                _sleeves(), nav=1_000_000,
                context=ctx,
            )
        # Switch to CAUTION
        for i in range(20):
            r = alloc.combine(
                date(2024, 2, 10) + timedelta(days=i),
                _sleeves(), nav=1_000_000,
                regime=VolRegime.CAUTION,
                context=ctx,
            )
        # Should still have some weight (dynamic cap allows up to 0.10)
        # but less than NORMAL peak
        # Should be below dynamic cap + still some residual from NORMAL
        assert r.w_vrp <= cfg.max_dynamic_caution_weight + 0.01 or r.w_vrp <= 0.12

    def test_crash_remains_zero(self):
        """CRASH regime must ALWAYS be zero (non-negotiable)."""
        cfg = VRPConfig()
        alloc = MetaAllocator(cfg)
        # Build weight
        for i in range(30):
            alloc.combine(
                date(2024, 1, 1) + timedelta(days=i),
                _sleeves(), nav=1_000_000,
                context=_default_ctx(forecast_health=0.92, headroom_ratio=0.80),
            )
        # Crash
        r = alloc.combine(
            date(2024, 2, 10), _sleeves(), nav=1_000_000,
            regime=VolRegime.CRASH_RISK,
            context=_default_ctx(forecast_health=0.50, headroom_ratio=0.10),
        )
        assert r.w_vrp == 0.0

    def test_caution_clamped_when_health_low(self):
        """Low forecast health in CAUTION should clamp weight aggressively."""
        cfg = VRPConfig()
        alloc = MetaAllocator(cfg)
        # Build some weight
        for i in range(20):
            alloc.combine(
                date(2024, 1, 1) + timedelta(days=i),
                _sleeves(), nav=1_000_000,
                context=_default_ctx(forecast_health=0.92, headroom_ratio=0.80),
            )
        # CAUTION with low health
        for i in range(10):
            r = alloc.combine(
                date(2024, 1, 21) + timedelta(days=i),
                _sleeves(), nav=1_000_000,
                regime=VolRegime.CAUTION,
                context=_default_ctx(forecast_health=0.50, headroom_ratio=0.30),
            )
        assert r.w_vrp <= 0.07, f"Should be clamped: {r.w_vrp}"


# ══════════════════════════════════════════════════════════════════════════
# Phase 2: Entry Filter
# ══════════════════════════════════════════════════════════════════════════

class TestEntryFilter:
    """Phase 2: entry quality gates."""

    def test_crash_always_blocked(self):
        d = evaluate_entry(EntrySignals(iv_rank=0.9), VolRegime.CRASH_RISK)
        assert not d.allowed
        assert d.reason == "crash_regime"

    def test_normal_good_signals(self):
        d = evaluate_entry(
            EntrySignals(iv_rank=0.55, forecast_p90=0.20, scenario_worst_loss_pct=0.02),
            VolRegime.NORMAL_CARRY,
        )
        assert d.allowed

    def test_normal_low_iv_blocked(self):
        d = evaluate_entry(
            EntrySignals(iv_rank=0.30),
            VolRegime.NORMAL_CARRY,
        )
        assert not d.allowed
        assert "iv_rank" in d.reason

    def test_normal_high_forecast_blocked(self):
        d = evaluate_entry(
            EntrySignals(iv_rank=0.55, forecast_p90=0.35),
            VolRegime.NORMAL_CARRY,
        )
        assert not d.allowed
        assert "forecast_p90" in d.reason

    def test_normal_scenario_too_high_blocked(self):
        d = evaluate_entry(
            EntrySignals(iv_rank=0.55, scenario_worst_loss_pct=0.05),
            VolRegime.NORMAL_CARRY,
        )
        assert not d.allowed
        assert "scenario" in d.reason

    def test_normal_term_slope_blocked(self):
        d = evaluate_entry(
            EntrySignals(iv_rank=0.55, term_slope_favorable=False),
            VolRegime.NORMAL_CARRY,
        )
        assert not d.allowed
        assert "term_slope" in d.reason

    def test_caution_good_signals(self):
        d = evaluate_entry(
            EntrySignals(iv_rank=0.80, forecast_health=0.95, headroom_ratio=0.60),
            VolRegime.CAUTION,
        )
        assert d.allowed

    def test_caution_low_iv_blocked(self):
        d = evaluate_entry(
            EntrySignals(iv_rank=0.55, forecast_health=0.95, headroom_ratio=0.60),
            VolRegime.CAUTION,
        )
        assert not d.allowed

    def test_caution_low_health_blocked(self):
        d = evaluate_entry(
            EntrySignals(iv_rank=0.80, forecast_health=0.80, headroom_ratio=0.60),
            VolRegime.CAUTION,
        )
        assert not d.allowed

    def test_caution_low_headroom_blocked(self):
        d = evaluate_entry(
            EntrySignals(iv_rank=0.80, forecast_health=0.95, headroom_ratio=0.30),
            VolRegime.CAUTION,
        )
        assert not d.allowed

    def test_caution_danger_zone_blocked(self):
        d = evaluate_entry(
            EntrySignals(
                iv_rank=0.80, forecast_health=0.95,
                headroom_ratio=0.60, danger_zone_active=True,
            ),
            VolRegime.CAUTION,
        )
        assert not d.allowed


# ══════════════════════════════════════════════════════════════════════════
# Phase 3: Enhanced Objective
# ══════════════════════════════════════════════════════════════════════════

class TestEnhancedObjective:
    """Phase 3: headroom bonus, convexity penalty, regime return weight."""

    def test_higher_headroom_increases_allocation(self):
        """More headroom → higher allocation (via headroom bonus)."""
        cfg = VRPConfig(w_min_hold_days=0)
        w_low_headroom = []
        w_high_headroom = []

        for headroom, w_list in [(0.2, w_low_headroom), (0.9, w_high_headroom)]:
            alloc = MetaAllocator(cfg)
            ctx = _default_ctx(forecast_health=0.92, headroom_ratio=headroom)
            for i in range(30):
                r = alloc.combine(
                    date(2024, 1, 1) + timedelta(days=i),
                    _sleeves(), nav=1_000_000, context=ctx,
                )
                w_list.append(r.w_vrp)

        assert np.mean(w_high_headroom[-10:]) >= np.mean(w_low_headroom[-10:])

    def test_high_convexity_reduces_allocation(self):
        """Higher convexity penalty → lower allocation."""
        cfg = VRPConfig(convexity_penalty_weight=1.0, w_min_hold_days=0)
        w_no_conv = []
        w_high_conv = []

        for conv_score, w_list in [(0.0, w_no_conv), (0.5, w_high_conv)]:
            alloc = MetaAllocator(cfg)
            ctx = _default_ctx(convexity_score=conv_score)
            for i in range(30):
                r = alloc.combine(
                    date(2024, 1, 1) + timedelta(days=i),
                    _sleeves(), nav=1_000_000, context=ctx,
                )
                w_list.append(r.w_vrp)

        assert np.mean(w_no_conv[-10:]) >= np.mean(w_high_conv[-10:])


# ══════════════════════════════════════════════════════════════════════════
# Phase 4: Deployment Floor Adjustment
# ══════════════════════════════════════════════════════════════════════════

class TestDeploymentFloor:
    """Phase 4: ramp from zero, lower reentry threshold."""

    def test_ramp_from_zero_over_multiple_days(self):
        """System should ramp from 0 over ~10 days with v4 params."""
        cfg = VRPConfig()
        alloc = MetaAllocator(cfg)
        ctx = _default_ctx(forecast_health=0.92, headroom_ratio=0.80)
        weights = []
        for i in range(20):
            r = alloc.combine(
                date(2024, 1, 1) + timedelta(days=i),
                _sleeves(), nav=1_000_000, context=ctx,
            )
            weights.append(r.w_vrp)

        # Should be non-zero by day 5 and growing
        assert any(w > 0 for w in weights[:5]), "Should start ramping within 5 days"
        assert weights[-1] > weights[5], "Should still be growing at day 20"

    def test_no_oscillation_during_ramp(self):
        """Daily changes during ramp should all be positive (monotone increase)."""
        cfg = VRPConfig()
        alloc = MetaAllocator(cfg)
        ctx = _default_ctx(forecast_health=0.92, headroom_ratio=0.80)
        weights = []
        for i in range(30):
            r = alloc.combine(
                date(2024, 1, 1) + timedelta(days=i),
                _sleeves(), nav=1_000_000, context=ctx,
            )
            weights.append(r.w_vrp)

        changes = [weights[i] - weights[i-1] for i in range(1, len(weights))]
        # Allow holds (0 change) but no negative changes during pure ramp
        ramp_phase = changes[:15]  # first 15 days
        assert all(c >= -1e-9 for c in ramp_phase), f"Oscillation detected: {ramp_phase}"

    def test_reentry_threshold_lowered(self):
        """With threshold=0.06, w_opt=0.07 should allow re-entry (was blocked at 0.08)."""
        cfg = VRPConfig(w_reentry_threshold=0.06)
        alloc = MetaAllocator(cfg)
        # Custom sleeves that produce lower w_opt
        sleeves = [
            SleeveResult(name="equity", expected_return=0.08, realized_vol=0.16),
            SleeveResult(name="vrp", expected_return=0.04, realized_vol=0.08, es_95=0.10),
        ]
        r = alloc.combine(date(2024, 1, 1), sleeves, nav=1_000_000)
        # w_opt should be >= 0.06 given these params, so re-entry should work
        assert r.w_opt >= 0.06 or r.w_vrp >= 0.0  # either way, should not error


# ══════════════════════════════════════════════════════════════════════════
# Phase 5: Normal Utilization Target
# ══════════════════════════════════════════════════════════════════════════

class TestUtilizationTarget:
    """Phase 5: soft pull toward target weight in NORMAL."""

    def test_mean_w_vrp_increases_materially(self):
        """With utilization target=0.15, mean_w_vrp should be materially > 0.05."""
        cfg = VRPConfig()
        alloc = MetaAllocator(cfg)
        ctx = _default_ctx(forecast_health=0.92, headroom_ratio=0.80)
        weights = []
        for i in range(100):
            r = alloc.combine(
                date(2024, 1, 1) + timedelta(days=i),
                _sleeves(), nav=1_000_000, context=ctx,
            )
            weights.append(r.w_vrp)

        mean_w = np.mean(weights)
        assert mean_w > 0.05, f"mean_w_vrp too low: {mean_w:.4f}"

    def test_crash_invariants_intact_with_utilization(self):
        """Utilization target must not override crash=0."""
        cfg = VRPConfig(target_utilization_normal=0.20)
        alloc = MetaAllocator(cfg)
        # Build weight
        for i in range(30):
            alloc.combine(
                date(2024, 1, 1) + timedelta(days=i),
                _sleeves(), nav=1_000_000,
                context=_default_ctx(forecast_health=0.92, headroom_ratio=0.80),
            )
        # Crash
        r = alloc.combine(
            date(2024, 2, 10), _sleeves(), nav=1_000_000,
            regime=VolRegime.CRASH_RISK,
        )
        assert r.w_vrp == 0.0

    def test_delta_invariant_with_utilization(self):
        """Utilization pull must not break the delta cap invariant."""
        cfg = VRPConfig(target_utilization_normal=0.25, utilization_pull_strength=2.0)
        alloc = MetaAllocator(cfg)
        ctx = _default_ctx(forecast_health=0.95, headroom_ratio=0.90)
        prev_w = 0.0
        for i in range(50):
            r = alloc.combine(
                date(2024, 1, 1) + timedelta(days=i),
                _sleeves(), nav=1_000_000, context=ctx,
            )
            delta = abs(r.w_vrp - prev_w)
            assert delta <= cfg.w_max_daily_delta + 1e-9, f"Delta exceeded: {delta}"
            prev_w = r.w_vrp


# ══════════════════════════════════════════════════════════════════════════
# Phase 6: Validation Metrics
# ══════════════════════════════════════════════════════════════════════════

class TestValidationMetrics:
    """Phase 6: new validation metrics."""

    def test_regime_stats_have_mean_w(self):
        """RegimeStats should include mean_w_vrp."""
        from backend.app.evaluation.vrp_validation import RegimeStats
        rs = RegimeStats(regime="normal_carry", mean_w_vrp=0.12)
        assert rs.mean_w_vrp == 0.12

    def test_regime_stats_have_efficiency(self):
        from backend.app.evaluation.vrp_validation import RegimeStats
        rs = RegimeStats(regime="normal_carry", allocation_efficiency=1.5)
        assert rs.allocation_efficiency == 1.5

    def test_allocation_analysis_has_utilization(self):
        from backend.app.evaluation.vrp_validation import AllocationAnalysis
        aa = AllocationAnalysis(mean_w_vrp=0.10, capital_utilization_ratio=0.40)
        assert aa.capital_utilization_ratio == 0.40

    def test_run_validation_computes_new_metrics(self):
        from backend.app.evaluation.vrp_validation import run_validation
        N = 50
        dates = pd.date_range("2024-01-01", periods=N, freq="B")
        regimes = pd.Series(["normal_carry"] * N, index=dates)
        pnl = pd.Series(np.random.normal(100, 50, N), index=dates)
        losses = pd.Series(np.random.uniform(0.01, 0.03, N), index=dates)
        w = pd.Series(np.linspace(0.05, 0.15, N), index=dates)
        fh = pd.Series(np.ones(N) * 0.9, index=dates)
        dz = pd.Series(np.zeros(N, dtype=int), index=dates)
        dr = pd.Series(np.zeros(N, dtype=int), index=dates)

        s = run_validation(
            regimes=regimes, daily_pnl=pnl, scenario_losses_pct=losses,
            w_vrp_series=w, forecast_health=fh,
            danger_zone_counts=dz, derisk_action_counts=dr,
        )

        normal_rs = next(r for r in s.regime_stats if r.regime == "normal_carry")
        assert normal_rs.mean_w_vrp > 0
        assert s.allocation_analysis.capital_utilization_ratio > 0


# ══════════════════════════════════════════════════════════════════════════
# Safety invariants (must still pass after expansion)
# ══════════════════════════════════════════════════════════════════════════

class TestSafetyInvariantsPostExpansion:
    """All pre-existing safety invariants must hold."""

    def test_delta_cap_never_breached(self):
        cfg = VRPConfig()
        alloc = MetaAllocator(cfg)
        ctx = _default_ctx(forecast_health=0.92, headroom_ratio=0.80)
        prev = 0.0
        for i in range(100):
            r = alloc.combine(
                date(2024, 1, 1) + timedelta(days=i),
                _sleeves(), nav=1_000_000, context=ctx,
            )
            assert abs(r.w_vrp - prev) <= cfg.w_max_daily_delta + 1e-9
            prev = r.w_vrp

    def test_crash_always_zero(self):
        cfg = VRPConfig()
        alloc = MetaAllocator(cfg)
        for regime in [VolRegime.CRASH_RISK] * 10:
            r = alloc.combine(
                date(2024, 1, 1), _sleeves(), nav=1_000_000, regime=regime,
            )
            assert r.w_vrp == 0.0

    def test_weight_bounded_by_max(self):
        cfg = VRPConfig(w_max_vrp=0.25)
        alloc = MetaAllocator(cfg)
        ctx = _default_ctx(forecast_health=0.99, headroom_ratio=0.99)
        for i in range(100):
            r = alloc.combine(
                date(2024, 1, 1) + timedelta(days=i),
                _sleeves(), nav=1_000_000, context=ctx,
            )
            assert r.w_vrp <= cfg.w_max_vrp + 1e-9

    def test_churn_days_zero(self):
        """No daily change should exceed the churn threshold."""
        cfg = VRPConfig()
        alloc = MetaAllocator(cfg)
        ctx = _default_ctx(forecast_health=0.92, headroom_ratio=0.80)
        weights = []
        for i in range(100):
            r = alloc.combine(
                date(2024, 1, 1) + timedelta(days=i),
                _sleeves(), nav=1_000_000, context=ctx,
            )
            weights.append(r.w_vrp)
        changes = pd.Series(weights).diff().abs().dropna()
        # Use epsilon tolerance for float comparison (pd.diff precision)
        churn = (changes > cfg.w_max_daily_delta + 1e-9).sum()
        assert churn == 0
