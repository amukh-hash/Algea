"""
De-risk policy and exit logic for the VRP sleeve.

v3: Action type accounting (close/tighten_stop/block_entry), liquidity deferral
    cap (max_optional_deferral_days), separate counters for churn metrics.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from algae.execution.options.config import VRPConfig
from algae.execution.options.structures import DerivativesPosition


# ═══════════════════════════════════════════════════════════════════════════
# Exit reason enum
# ═══════════════════════════════════════════════════════════════════════════

class ExitReason(enum.Enum):
    DTE_EXIT = "dte_exit"
    PROFIT_TAKE = "profit_take"
    STOP_LOSS = "stop_loss"
    REGIME_DERISK = "regime_derisk"
    DANGER_ZONE = "danger_zone"
    SCENARIO_BUDGET = "scenario_budget"
    LIQUIDITY = "liquidity"
    EARLY_ASSIGNMENT = "early_assignment"
    LIQUIDITY_DEFERRAL_FORCED = "liquidity_deferral_forced"


# ═══════════════════════════════════════════════════════════════════════════
# Action type — distinguish close vs tighten vs block
# ═══════════════════════════════════════════════════════════════════════════

class ActionType(enum.Enum):
    CLOSE = "close"
    TIGHTEN_STOP = "tighten_stop"
    BLOCK_ENTRY = "block_entry"


# ═══════════════════════════════════════════════════════════════════════════
# De-risk action
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DeRiskAction:
    """A single prioritised close recommendation."""
    position_id: str
    position: DerivativesPosition
    reason: ExitReason
    priority_score: float          # higher → close first
    scenario_contribution: float   # $ loss in worst-case scenario
    required_to_restore_limits: bool = False
    action_type: ActionType = ActionType.CLOSE


# ═══════════════════════════════════════════════════════════════════════════
# De-risk summary (v3 — split counters)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DeRiskSummary:
    """Audit-friendly summary of the de-risk evaluation."""
    regime: str
    total_scenario_loss: float
    scenario_budget: float
    positions_evaluated: int
    actions: List[DeRiskAction] = field(default_factory=list)
    constraints_restored: bool = False
    remaining_scenario_loss: float = 0.0

    @property
    def close_count(self) -> int:
        return sum(1 for a in self.actions if a.action_type == ActionType.CLOSE)

    @property
    def tighten_count(self) -> int:
        return sum(1 for a in self.actions if a.action_type == ActionType.TIGHTEN_STOP)

    @property
    def blocked_count(self) -> int:
        return sum(1 for a in self.actions if a.action_type == ActionType.BLOCK_ENTRY)


# ═══════════════════════════════════════════════════════════════════════════
# De-risk policy (v3 — deferral tracking, action types)
# ═══════════════════════════════════════════════════════════════════════════

class DeRiskPolicy:
    """Constraint-driven prioritised de-risk policy.

    v3 additions:
    - Tracks liquidity deferral days per position.
    - Forces close after max_optional_deferral_days.
    - All actions tagged with ActionType for clean accounting.
    """

    def __init__(self, config: VRPConfig) -> None:
        self.config = config
        # Track consecutive days each position has been deferred
        self._deferral_days: Dict[str, int] = {}

    def evaluate(
        self,
        positions: List[DerivativesPosition],
        regime: str,
        scenario_contributions: Dict[str, float],
        total_scenario_loss: float,
        nav: float,
        danger_zone_flags: Optional[Dict[str, bool]] = None,
        current_spreads: Optional[Dict[str, float]] = None,
    ) -> DeRiskSummary:
        """Evaluate which positions to close and in what order.

        Parameters
        ----------
        positions : open positions
        regime : "normal_carry" | "caution" | "crash_risk"
        scenario_contributions : {position_id: worst-case $ loss}
        total_scenario_loss : aggregate worst-case $ loss
        nav : current AUM
        danger_zone_flags : {position_id: True if in danger zone}
        current_spreads : {position_id: bid_ask_spread_pct} for liquidity gate
        
        Returns
        -------
        DeRiskSummary with ordered close actions.
        """
        cfg = self.config
        dz = danger_zone_flags or {}
        spreads = current_spreads or {}
        scenario_budget = cfg.max_worst_case_scenario_loss_pct_nav * nav

        summary = DeRiskSummary(
            regime=regime,
            total_scenario_loss=total_scenario_loss,
            scenario_budget=scenario_budget,
            positions_evaluated=len(positions),
        )

        # Normalise: compare absolute loss against budget (both positive)
        abs_loss = abs(total_scenario_loss)

        # Nothing to do if constraints satisfied and no danger-zone positions
        if regime != "crash_risk" and abs_loss <= scenario_budget:
            any_danger = any(dz.get(p.position_id, False) for p in positions)
            if not any_danger:
                summary.constraints_restored = True
                summary.remaining_scenario_loss = total_scenario_loss
                # Reset deferral days for positions no longer needing close
                self._deferral_days.clear()
                return summary

        # Score each position — higher = close first
        scored: List[Tuple[float, DerivativesPosition]] = []
        for pos in positions:
            sc = abs(scenario_contributions.get(pos.position_id, 0.0))
            in_dz = dz.get(pos.position_id, False)
            from datetime import date as _date
            dte = max((pos.expiry - _date.today()).days, 0)
            dte_proximity = max(1.0 - dte / 30.0, 0.0)
            score = sc + (500.0 if in_dz else 0.0) + dte_proximity * 100.0
            scored.append((score, pos))

        scored.sort(key=lambda t: t[0], reverse=True)

        # Close positions until constraints restored or cap reached
        running_abs_loss = abs_loss
        actions_taken = 0

        for score, pos in scored:
            if actions_taken >= cfg.max_daily_derisk_actions:
                break

            sc = abs(scenario_contributions.get(pos.position_id, 0.0))
            in_dz = dz.get(pos.position_id, False)

            # Determine if required to restore limits
            required_for_limits = (
                running_abs_loss > scenario_budget
                or in_dz
                or regime == "crash_risk"
            )

            if not required_for_limits:
                break  # constraints satisfied, stop closing

            # Liquidity gate with deferral cap
            pos_spread = spreads.get(pos.position_id, 0.0)
            is_truly_required = (
                running_abs_loss > scenario_budget
                or in_dz
                or regime == "crash_risk"
            )
            deferral_count = self._deferral_days.get(pos.position_id, 0)
            deferral_forced = deferral_count >= cfg.max_optional_deferral_days

            if (cfg.liquidity_block_optional_closes
                    and pos_spread > cfg.max_spread_pct_live
                    and not is_truly_required
                    and not deferral_forced):
                # Defer — liquidity too thin for optional close
                self._deferral_days[pos.position_id] = deferral_count + 1
                continue

            # If deferral forced, use special reason
            reason = ExitReason.LIQUIDITY_DEFERRAL_FORCED if deferral_forced and pos_spread > cfg.max_spread_pct_live else (
                ExitReason.DANGER_ZONE if in_dz
                else ExitReason.SCENARIO_BUDGET if running_abs_loss > scenario_budget
                else ExitReason.REGIME_DERISK
            )

            action = DeRiskAction(
                position_id=pos.position_id,
                position=pos,
                reason=reason,
                priority_score=score,
                scenario_contribution=sc,
                required_to_restore_limits=running_abs_loss > scenario_budget,
                action_type=ActionType.CLOSE,
            )
            summary.actions.append(action)
            running_abs_loss -= sc
            actions_taken += 1

            # Clear deferral tracker on close
            self._deferral_days.pop(pos.position_id, None)

            # Check if constraints now satisfied
            if cfg.close_until_constraints_satisfied and running_abs_loss <= scenario_budget:
                if regime != "crash_risk":
                    break

        summary.constraints_restored = running_abs_loss <= scenario_budget
        summary.remaining_scenario_loss = total_scenario_loss + (abs_loss - running_abs_loss)
        return summary
