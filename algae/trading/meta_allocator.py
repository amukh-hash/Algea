"""
MetaAllocator — combines equity + VRP sleeves under global constraints.

v4: Controlled exposure expansion with preserved safety invariants.

Pipeline: w_opt → regime_cap(dynamic) → smooth → delta_cap → hold → floor → w_final

New in v4:
  - Dynamic caution cap: min(0.50 * w_opt, max_dynamic_caution_weight)
  - Normal boost multiplier when forecast_health + headroom strong
  - Enhanced objective with headroom bonus + utilization pull + convexity penalty
  - Faster ramp via alpha=0.18, reentry=0.06

Invariants PRESERVED:
  - |Δw_final| ≤ w_max_daily_delta (unless CRASH_RISK)
  - CRASH_RISK → w_vrp = 0 immediately (never weakened)
  - Weight floor snap-to-zero on decay only
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from algae.data.options.vrp_features import VolRegime
from algae.execution.options.config import VRPConfig


# ═══════════════════════════════════════════════════════════════════════════
# Sleeve result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SleeveResult:
    """Summary of a single sleeve's risk/return characteristics."""
    name: str
    expected_return: float
    realized_vol: float
    max_drawdown: float = 0.0
    es_95: float = 0.0
    es_99: float = 0.0
    forecast_risk: Optional[Dict[float, float]] = None   # {0.95: rv_val, ...}


# ═══════════════════════════════════════════════════════════════════════════
# Allocation context (optional signals for enhanced objective)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AllocationContext:
    """Optional context signals passed to the allocator."""
    forecast_health: float = 1.0
    headroom_ratio: float = 1.0       # headroom / budget  (1.0 = full budget available)
    scenario_worst_loss_pct: float = 0.0
    convexity_score: float = 0.0
    forecast_p90: Optional[float] = None
    forecast_p95: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════════
# Allocation result (v4 — full pipeline visibility)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AllocationResult:
    """Output of the MetaAllocator.combine() method."""
    as_of_date: date
    w_equity: float
    w_vrp: float                       # alias for w_final
    tail_adjusted_sharpe: float
    regime: str
    risk_budget: Dict[str, float]
    # v3/v4 pipeline stages
    w_opt: float = 0.0                 # raw grid search output
    w_regime_cap: float = 0.0          # after regime multiplier
    w_smoothed: float = 0.0            # after EMA
    w_delta_capped: float = 0.0        # after daily delta cap
    w_final: float = 0.0               # final weight (= w_vrp)
    # compat
    w_vrp_raw: float = 0.0
    w_vrp_prev: float = 0.0
    smoothing_applied: bool = False
    hold_days_remaining: int = 0


# ═══════════════════════════════════════════════════════════════════════════
# Allocator state (persisted across days)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AllocatorState:
    """Tracks allocator state for smoothing + hold logic."""
    w_vrp_prev: float = 0.0
    last_change_date: Optional[date] = None
    hold_days_used: int = 0
    weight_history: List[float] = field(default_factory=list)

    def weight_volatility(self, window: int = 20) -> float:
        """Std of recent weight changes (uses w_final history only)."""
        if len(self.weight_history) < 3:
            return 0.0
        tail = self.weight_history[-window:]
        changes = [abs(tail[i] - tail[i - 1]) for i in range(1, len(tail))]
        return float(np.std(changes)) if len(changes) > 1 else 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Allocator invariant error
# ═══════════════════════════════════════════════════════════════════════════

class AllocatorInvariantError(RuntimeError):
    """Raised when the allocator violates its daily delta invariant."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# MetaAllocator v4
# ═══════════════════════════════════════════════════════════════════════════

class MetaAllocator:
    """Optimise equity/VRP weight via tail-adjusted Sharpe grid search.

    v4 pipeline:
      w_opt → regime_cap(dynamic) → smooth → delta_cap → hold → floor → w_final

    Safety invariants (NEVER weakened):
      - |w_final[t] - w_final[t-1]| <= w_max_daily_delta
      - CRASH_RISK → w_vrp = 0 immediately
      - w_vrp ∈ [0, w_max_vrp]
    """

    def __init__(
        self,
        config: VRPConfig | None = None,
        state: AllocatorState | None = None,
    ) -> None:
        self.config = config or VRPConfig()
        self.state = state or AllocatorState()

    def _dynamic_regime_cap(
        self,
        regime: VolRegime,
        w_opt: float,
        ctx: AllocationContext,
    ) -> float:
        """Compute dynamic regime cap.

        NORMAL: up to w_opt. Boost 1.2x if health+headroom good.
        CAUTION: dynamic cap = min(caution_mult * w_opt, max_dynamic_caution_weight)
        CRASH: always 0 — non-negotiable.
        """
        cfg = self.config

        if regime == VolRegime.CRASH_RISK:
            return 0.0  # NEVER CHANGE

        if regime == VolRegime.CAUTION:
            if cfg.enable_dynamic_caution_cap:
                # Dynamic: up to 50% of w_opt, hard-capped at max_dynamic_caution_weight
                cap = min(cfg.w_vrp_regime_cap_caution * w_opt, cfg.max_dynamic_caution_weight)
                # But if forecast health is below threshold, clamp aggressively
                if ctx.forecast_health < cfg.min_forecast_health:
                    cap = min(cap, 0.02)
                return cap
            else:
                return cfg.w_vrp_regime_cap_caution * w_opt

        # NORMAL_CARRY
        cap = cfg.w_vrp_regime_cap_normal * w_opt
        # Boost if both forecast health and headroom are strong
        if (ctx.forecast_health > 0.85
                and ctx.headroom_ratio > cfg.min_headroom_normal_boost):
            cap = min(cap * cfg.normal_boost_multiplier, cfg.w_max_vrp)
        return cap

    def combine(
        self,
        as_of_date: date,
        sleeves: List[SleeveResult],
        nav: float = 1_000_000,
        rf: float = 0.05,
        regime: Optional[VolRegime] = None,
        context: Optional[AllocationContext] = None,
    ) -> AllocationResult:
        """Find optimal equity/VRP weight under enhanced tail-penalised objective.

        Parameters
        ----------
        as_of_date : date for this allocation decision
        sleeves : list of SleeveResult (one per sleeve)
        nav : current AUM
        rf : risk-free rate
        regime : if provided, overrides regime detection
        context : optional AllocationContext with health/headroom/forecast signals
        """
        cfg = self.config
        ctx = context or AllocationContext()

        equity_sleeve = next((s for s in sleeves if s.name == "equity"), None)
        vrp_sleeve = next((s for s in sleeves if s.name == "vrp"), None)

        if regime is None:
            regime = VolRegime.NORMAL_CARRY

        w_prev = self.state.w_vrp_prev

        # --- CRASH_RISK → w_vrp = 0 immediately (INVARIANT — never weakened) ---
        if regime == VolRegime.CRASH_RISK:
            self.state.w_vrp_prev = 0.0
            self.state.last_change_date = as_of_date
            self.state.weight_history.append(0.0)
            return AllocationResult(
                as_of_date=as_of_date,
                w_equity=1.0,
                w_vrp=0.0,
                tail_adjusted_sharpe=0.0,
                regime=regime.value,
                risk_budget={"equity": 1.0, "vrp": 0.0},
                w_opt=0.0, w_regime_cap=0.0, w_smoothed=0.0,
                w_delta_capped=0.0, w_final=0.0,
                w_vrp_raw=0.0, w_vrp_prev=w_prev,
                smoothing_applied=False,
            )

        # --- No VRP sleeve ---
        if vrp_sleeve is None:
            return AllocationResult(
                as_of_date=as_of_date,
                w_equity=1.0, w_vrp=0.0,
                tail_adjusted_sharpe=0.0,
                regime=regime.value,
                risk_budget={"equity": 1.0, "vrp": 0.0},
            )

        # --- Estimate parameters ---
        eq_ret = equity_sleeve.expected_return if equity_sleeve else 0.08
        eq_vol = equity_sleeve.realized_vol if equity_sleeve else 0.16
        vrp_ret = vrp_sleeve.expected_return
        vrp_vol = vrp_sleeve.realized_vol

        # Tail estimate
        historical_es95 = vrp_sleeve.es_95 if vrp_sleeve.es_95 > 0 else vrp_vol * 1.65
        forecast_es95 = historical_es95
        if vrp_sleeve.forecast_risk:
            rv_p95 = vrp_sleeve.forecast_risk.get(0.95, None)
            if rv_p95 is not None:
                forecast_es95 = rv_p95 * 1.65
        tail_estimate = max(historical_es95, forecast_es95)

        # --- Regime-weighted return (Phase 3 objective enhancement) ---
        regime_ret_weight = (
            cfg.regime_return_weight_normal if regime == VolRegime.NORMAL_CARRY
            else cfg.regime_return_weight_caution
        )

        # --- Grid search for w_opt ---
        lam = cfg.tail_penalty_lambda
        m = cfg.dd_penalty_m
        step = cfg.allocator_grid_step
        w_max = cfg.w_max_vrp

        # Convexity penalty from context
        conv_penalty = cfg.convexity_penalty_weight * ctx.convexity_score

        best_w = 0.0
        best_obj = -np.inf
        w = 0.0
        while w <= w_max + 1e-9:
            w_eq = 1.0 - w
            port_ret = w_eq * eq_ret + w * vrp_ret * regime_ret_weight - rf
            port_vol = np.sqrt((w_eq * eq_vol) ** 2 + (w * vrp_vol) ** 2)
            tail_adj_vol = max(port_vol, lam * w * tail_estimate)
            sharpe = port_ret / max(tail_adj_vol, 1e-6)

            vrp_dd = vrp_sleeve.max_drawdown if vrp_sleeve.max_drawdown > 0 else 0.0
            dd_penalty = m * w * vrp_dd

            # Headroom bonus (Phase 3): reward allocation when headroom is ample
            headroom_bonus = cfg.headroom_bonus_weight * w * max(ctx.headroom_ratio - 0.3, 0.0)

            # Utilization pull (Phase 5): soft bias toward target in NORMAL
            util_pull = 0.0
            if regime == VolRegime.NORMAL_CARRY and ctx.forecast_health > cfg.min_forecast_health:
                target = cfg.target_utilization_normal
                gap = target - w
                if gap > 0:
                    util_pull = cfg.utilization_pull_strength * gap * 0.1

            obj = sharpe - dd_penalty - conv_penalty * w + headroom_bonus + util_pull

            if obj > best_obj:
                best_obj = obj
                best_w = w

            w += step

        w_opt = float(round(best_w, 4))

        # Weight volatility penalty
        w_vol = self.state.weight_volatility(cfg.w_history_window)
        if w_vol > 0 and cfg.w_volatility_penalty > 0:
            best_obj -= cfg.w_volatility_penalty * w_vol

        # ============================================================
        # v4 PIPELINE: w_opt → dynamic_regime_cap → smooth → cap → hold → floor → final
        # ============================================================

        # Step 1: Dynamic regime cap (BEFORE smoothing)
        w_regime_cap = self._dynamic_regime_cap(regime, w_opt, ctx)

        # Step 2: Re-entry hysteresis
        if w_prev == 0.0 and w_opt < cfg.w_reentry_threshold:
            w_regime_cap = 0.0

        # Step 3: EMA smoothing
        w_smoothed = cfg.w_smoothing_alpha * w_regime_cap + (1.0 - cfg.w_smoothing_alpha) * w_prev

        # Step 4: Daily delta cap (INVARIANT)
        delta = w_smoothed - w_prev
        if abs(delta) > cfg.w_max_daily_delta:
            w_smoothed = w_prev + np.sign(delta) * cfg.w_max_daily_delta
        w_capped = round(w_smoothed, 10)  # avoid float drift

        # Step 5: Min hold days — only suppress DIRECTION REVERSALS
        #   During monotone ramp-up, each +0.02 step is same direction → allow through.
        #   Only freeze weight when the new delta would flip sign from last change.
        hold_remaining = 0
        if self.state.last_change_date is not None:
            days_since = (as_of_date - self.state.last_change_date).days
            if days_since < cfg.w_min_hold_days and abs(w_capped - w_prev) > 0.001:
                # Check if this is a direction reversal
                last_dir = w_prev - (self.state.weight_history[-2] if len(self.state.weight_history) >= 2 else w_prev)
                new_dir = w_capped - w_prev
                is_reversal = (last_dir > 0.001 and new_dir < -0.001) or (last_dir < -0.001 and new_dir > 0.001)
                if is_reversal:
                    w_capped = w_prev  # hold current weight
                    hold_remaining = cfg.w_min_hold_days - days_since

        # Step 6: Clip to [0, w_max]
        w_capped = float(np.clip(w_capped, 0.0, w_max))

        # Step 7: Weight floor — snap tiny non-zero to zero on DECAY only
        #   Rules:
        #   - Only snap to zero when weight is DECREASING (not during ramp-up)
        #   - NEVER snap if the resulting delta would violate the cap invariant
        #   - During ramp-up (w_capped > w_prev), allow through even if < floor
        if 0 < w_capped < cfg.w_min_deployment:
            increasing = w_capped > w_prev + 1e-9
            snap_delta = abs(0.0 - w_prev)
            snap_safe = snap_delta <= cfg.w_max_daily_delta + 1e-9
            if not increasing and snap_safe:
                w_capped = 0.0
            # else: keep w_capped (either ramping up or snap would breach delta)

        w_final = round(float(w_capped), 10)

        # --- Invariant check (NEVER removed) ---
        actual_delta = abs(w_final - w_prev)
        if actual_delta > cfg.w_max_daily_delta + 1e-9 and regime != VolRegime.CRASH_RISK:
            raise AllocatorInvariantError(
                f"Δw_final = {actual_delta:.4f} exceeds cap {cfg.w_max_daily_delta} "
                f"(w_prev={w_prev:.4f}, w_final={w_final:.4f}, regime={regime.value})"
            )

        # --- Update state (w_final ONLY) ---
        smoothing_applied = abs(w_final - w_opt) > 0.001
        if abs(w_final - w_prev) > 0.001:
            self.state.last_change_date = as_of_date
        self.state.w_vrp_prev = w_final
        self.state.weight_history.append(w_final)

        w_equity = 1.0 - w_final

        return AllocationResult(
            as_of_date=as_of_date,
            w_equity=w_equity,
            w_vrp=w_final,
            tail_adjusted_sharpe=best_obj,
            regime=regime.value,
            risk_budget={"equity": w_equity, "vrp": w_final},
            w_opt=w_opt,
            w_regime_cap=float(round(w_regime_cap, 4)),
            w_smoothed=float(round(w_smoothed, 6)),
            w_delta_capped=float(round(w_capped, 6)),
            w_final=w_final,
            w_vrp_raw=w_opt,
            w_vrp_prev=w_prev,
            smoothing_applied=smoothing_applied,
            hold_days_remaining=hold_remaining,
        )
