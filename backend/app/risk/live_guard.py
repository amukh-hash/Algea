"""
Paper / live trading safety guard.

Blocks new trades and optionally reduces VRP allocation when
extreme risk conditions are detected.  All decisions are logged
for audit.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from algaie.execution.options.config import VRPConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Decision output
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class LiveGuardDecision:
    """Output of the live guard evaluation."""
    as_of_date: str
    allow_new_trades: bool = True
    reduce_vrp_pct: float = 0.0   # 0 = no reduction, 0.5 = cut by 50%
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ═══════════════════════════════════════════════════════════════════════════
# Live guard
# ═══════════════════════════════════════════════════════════════════════════

class LiveGuard:
    """Safety layer for paper and live trading.

    Checks:
    1. Scenario loss > hard limit → block new trades
    2. Margin utilisation > limit → block new trades
    3. Forecast health sudden drop → reduce VRP weight
    4. Regime is CRASH_RISK → block new trades
    """

    def __init__(self, config: Optional[VRPConfig] = None) -> None:
        self.config = config or VRPConfig()
        self._health_history: List[float] = []

    def evaluate(
        self,
        as_of_date: date,
        scenario_loss_pct: float,
        margin_utilization: float,
        forecast_health: float,
        regime: str = "normal_carry",
    ) -> LiveGuardDecision:
        """Evaluate all safety conditions.

        Parameters
        ----------
        as_of_date : current date
        scenario_loss_pct : current worst-case scenario loss as % of NAV
        margin_utilization : current margin utilisation (0-1)
        forecast_health : current forecast health score (0-1)
        regime : current regime string
        """
        cfg = self.config
        decision = LiveGuardDecision(as_of_date=str(as_of_date))
        decision.metrics = {
            "scenario_loss_pct": scenario_loss_pct,
            "margin_utilization": margin_utilization,
            "forecast_health": forecast_health,
            "regime": regime,
        }

        # 1. Scenario loss hard limit
        if scenario_loss_pct > cfg.live_guard_hard_loss_limit:
            decision.allow_new_trades = False
            decision.reasons.append(
                f"scenario_loss_pct {scenario_loss_pct:.4f} > hard limit {cfg.live_guard_hard_loss_limit}"
            )

        # 2. Margin utilisation
        if margin_utilization > cfg.live_guard_margin_limit:
            decision.allow_new_trades = False
            decision.reasons.append(
                f"margin_utilization {margin_utilization:.2%} > limit {cfg.live_guard_margin_limit:.2%}"
            )

        # 3. Forecast health sudden drop
        self._health_history.append(forecast_health)
        if len(self._health_history) >= 4:
            health_3d_ago = self._health_history[-4]
            drop = health_3d_ago - forecast_health
            if drop > cfg.live_guard_health_drop_threshold:
                decision.reduce_vrp_pct = cfg.live_guard_health_drop_reduce_pct
                decision.reasons.append(
                    f"forecast health dropped {drop:.2f} in 3 days "
                    f"(from {health_3d_ago:.2f} to {forecast_health:.2f})"
                )

        # 4. Regime CRASH_RISK
        if regime == "crash_risk":
            decision.allow_new_trades = False
            decision.reasons.append("regime is CRASH_RISK")

        if decision.reasons:
            logger.warning(
                "LiveGuard %s: allow_trades=%s reduce=%.0f%% reasons=%s",
                as_of_date, decision.allow_new_trades,
                decision.reduce_vrp_pct * 100, decision.reasons,
            )

        return decision
