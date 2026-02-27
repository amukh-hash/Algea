"""
Paper / live trading safety guard.
"""
from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Dict, List, Optional

from algaie.execution.options.config import VRPConfig
from backend.app.ml_platform.drift.calibration import expected_calibration_error
from backend.app.ml_platform.drift.detectors import confidence_entropy_correlation, prediction_consistency_score

logger = logging.getLogger(__name__)


@dataclass
class LiveGuardDecision:
    as_of_date: str
    status: str = "ok"
    allow_new_trades: bool = True
    reduce_vrp_pct: float = 0.0
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class LiveGuard:
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
        current_confidence: float = 0.7,
        current_accuracy: float = 0.7,
        prediction_series: Optional[list[float]] = None,
        consistency_baseline: Optional[dict] = None,
        confidence_series: Optional[list[float]] = None,
        outcome_series: Optional[list[float]] = None,
        corr_baseline: float = 0.2,
        sla_breach: bool = False,
    ) -> LiveGuardDecision:
        cfg = self.config
        decision = LiveGuardDecision(as_of_date=str(as_of_date))
        prediction_series = prediction_series or []
        consistency_baseline = consistency_baseline or {}
        confidence_series = confidence_series or []
        outcome_series = outcome_series or []

        ece = expected_calibration_error(current_confidence, current_accuracy)
        consistency = prediction_consistency_score(prediction_series, consistency_baseline)
        corr = confidence_entropy_correlation(confidence_series, outcome_series)

        decision.metrics = {
            "scenario_loss_pct": scenario_loss_pct,
            "margin_utilization": margin_utilization,
            "forecast_health": forecast_health,
            "regime": regime,
            "ece": ece,
            "prediction_consistency_score": consistency,
            "conf_entropy_corr": corr,
            "interval_coverage_error": ece,
            "sla_breach": bool(sla_breach),
        }

        if scenario_loss_pct > cfg.live_guard_hard_loss_limit:
            decision.allow_new_trades = False
            decision.status = "halted"
            decision.reasons.append(f"scenario_loss_pct {scenario_loss_pct:.4f} > hard limit {cfg.live_guard_hard_loss_limit}")
            decision.reasons.append("halt_scenario_loss")

        if margin_utilization > cfg.live_guard_margin_limit:
            decision.allow_new_trades = False
            decision.status = "halted"
            decision.reasons.append(f"margin_utilization {margin_utilization:.2%} > limit {cfg.live_guard_margin_limit:.2%}")
            decision.reasons.append("halt_margin")

        self._health_history.append(forecast_health)
        if len(self._health_history) >= 4:
            health_3d_ago = self._health_history[-4]
            drop = health_3d_ago - forecast_health
            if drop > cfg.live_guard_health_drop_threshold:
                decision.reduce_vrp_pct = cfg.live_guard_health_drop_reduce_pct
                decision.reasons.append(
                    f"forecast health dropped {drop:.2f} in 3 days (from {health_3d_ago:.2f} to {forecast_health:.2f})"
                )
                decision.reasons.append("reduce_forecast_health_drop")

        if regime == "crash_risk":
            decision.allow_new_trades = False
            decision.status = "halted"
            decision.reasons.append("regime is CRASH_RISK")
            decision.reasons.append("halt_regime_crash_risk")

        ece_threshold = float(getattr(cfg, "live_guard_ece_threshold", 0.12))
        consistency_threshold = float(getattr(cfg, "live_guard_prediction_consistency_threshold", 2.5))
        corr_deviation_threshold = float(getattr(cfg, "live_guard_conf_entropy_corr_delta_threshold", 0.4))

        if ece > ece_threshold:
            decision.allow_new_trades = False
            decision.status = "halted"
            decision.reasons.append("halt_ece_spike")
        if consistency > consistency_threshold:
            decision.allow_new_trades = False
            decision.status = "halted"
            decision.reasons.append("halt_prediction_consistency")
        if abs(corr - corr_baseline) > corr_deviation_threshold:
            decision.allow_new_trades = False
            decision.status = "halted"
            decision.reasons.append("halt_conf_entropy_corr")

        if decision.metrics.get("sla_breach", False):
            decision.allow_new_trades = False
            decision.status = "halted"
            decision.reasons.append("halt_sla_breach")

        if decision.reasons:
            logger.warning(
                "LiveGuard %s: allow_trades=%s reduce=%.0f%% reasons=%s",
                as_of_date,
                decision.allow_new_trades,
                decision.reduce_vrp_pct * 100,
                decision.reasons,
            )

        return decision
