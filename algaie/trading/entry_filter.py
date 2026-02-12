"""
Entry filter — quality gates for VRP position entries.

Controls which market conditions permit new position entries.
NORMAL entries require moderate quality gates.
CAUTION entries require very high quality gates.
CRASH entries are always blocked.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from algaie.execution.options.config import VRPConfig
from algaie.data.options.vrp_features import VolRegime


@dataclass
class EntrySignals:
    """Market signals evaluated at entry time."""
    iv_rank: float = 0.0
    term_slope_favorable: bool = True
    forecast_p90: Optional[float] = None
    forecast_health: float = 1.0
    scenario_worst_loss_pct: float = 0.0
    headroom_ratio: float = 1.0          # headroom / budget
    danger_zone_active: bool = False


@dataclass
class EntryDecision:
    """Result of entry filter evaluation."""
    allowed: bool
    reason: str = ""


def evaluate_entry(
    signals: EntrySignals,
    regime: VolRegime,
    config: Optional[VRPConfig] = None,
) -> EntryDecision:
    """Evaluate whether a new VRP entry is allowed.

    Returns
    -------
    EntryDecision with allowed flag and reason string.
    """
    cfg = config or VRPConfig()

    # CRASH: always blocked
    if regime == VolRegime.CRASH_RISK:
        return EntryDecision(allowed=False, reason="crash_regime")

    budget = cfg.max_worst_case_scenario_loss_pct_nav

    if regime == VolRegime.CAUTION:
        # Very strict gates in CAUTION
        if signals.iv_rank < cfg.entry_iv_rank_caution:
            return EntryDecision(
                allowed=False,
                reason=f"caution_iv_rank_low ({signals.iv_rank:.2f} < {cfg.entry_iv_rank_caution})",
            )
        if signals.forecast_health < cfg.entry_min_health_caution:
            return EntryDecision(
                allowed=False,
                reason=f"caution_health_low ({signals.forecast_health:.2f} < {cfg.entry_min_health_caution})",
            )
        if signals.headroom_ratio < cfg.entry_min_headroom_caution:
            return EntryDecision(
                allowed=False,
                reason=f"caution_headroom_low ({signals.headroom_ratio:.2f} < {cfg.entry_min_headroom_caution})",
            )
        if signals.danger_zone_active:
            return EntryDecision(allowed=False, reason="caution_danger_zone_active")
        return EntryDecision(allowed=True, reason="caution_all_gates_passed")

    # NORMAL_CARRY
    if signals.iv_rank < cfg.entry_iv_rank_normal:
        return EntryDecision(
            allowed=False,
            reason=f"normal_iv_rank_low ({signals.iv_rank:.2f} < {cfg.entry_iv_rank_normal})",
        )

    if not signals.term_slope_favorable:
        return EntryDecision(allowed=False, reason="normal_term_slope_unfavorable")

    if signals.forecast_p90 is not None and signals.forecast_p90 > cfg.entry_forecast_p90_normal:
        return EntryDecision(
            allowed=False,
            reason=f"normal_forecast_p90_high ({signals.forecast_p90:.2f} > {cfg.entry_forecast_p90_normal})",
        )

    scenario_pct = signals.scenario_worst_loss_pct / budget if budget > 0 else 0
    if scenario_pct > cfg.entry_scenario_pct_normal:
        return EntryDecision(
            allowed=False,
            reason=f"normal_scenario_too_high ({scenario_pct:.1%} > {cfg.entry_scenario_pct_normal:.0%})",
        )

    return EntryDecision(allowed=True, reason="normal_all_gates_passed")
