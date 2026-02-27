"""
VRP daily telemetry and threshold monitoring.

Captures a daily snapshot of all risk surfaces, logs warnings when
thresholds are approached, and persists to CSV + JSON.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from algea.execution.options.config import VRPConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Alert
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class MonitorAlert:
    """A single threshold warning."""
    metric: str
    value: float
    threshold: float
    severity: str = "WARNING"  # WARNING | CRITICAL

    @property
    def message(self) -> str:
        return f"[{self.severity}] {self.metric}: {self.value:.4f} (threshold: {self.threshold:.4f})"


# ═══════════════════════════════════════════════════════════════════════════
# Daily snapshot
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class DailySnapshot:
    """Complete daily risk surface snapshot."""
    as_of_date: str
    regime: str = "normal_carry"
    forecast_p50: float = 0.0
    forecast_p90: float = 0.0
    forecast_p95: float = 0.0
    forecast_p99: float = 0.0
    forecast_health: float = 1.0
    scenario_worst_loss_pct: float = 0.0
    gamma_exposure: float = 0.0
    vega_exposure: float = 0.0
    short_convexity_score: float = 0.0
    w_vrp: float = 0.0
    margin_utilization: float = 0.0
    avg_spread_pct: float = 0.0
    derisk_actions: int = 0
    danger_zone_triggers: int = 0
    alerts: List[MonitorAlert] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["alerts"] = [asdict(a) for a in self.alerts]
        return d


# ═══════════════════════════════════════════════════════════════════════════
# Monitor
# ═══════════════════════════════════════════════════════════════════════════

class VRPMonitor:
    """Daily telemetry collector and threshold checker."""

    def __init__(self, config: Optional[VRPConfig] = None) -> None:
        self.config = config or VRPConfig()
        self._history: List[DailySnapshot] = []

    def record(self, snapshot: DailySnapshot) -> DailySnapshot:
        """Record a snapshot and check thresholds, appending alerts."""
        alerts = self._check_thresholds(snapshot)
        snapshot.alerts = alerts
        for alert in alerts:
            logger.warning(alert.message)
        self._history.append(snapshot)
        return snapshot

    def _check_thresholds(self, snap: DailySnapshot) -> List[MonitorAlert]:
        cfg = self.config
        alerts: List[MonitorAlert] = []

        # Scenario loss approaching budget
        budget = cfg.max_worst_case_scenario_loss_pct_nav
        if snap.scenario_worst_loss_pct > budget * cfg.monitor_scenario_warning_pct:
            sev = "CRITICAL" if snap.scenario_worst_loss_pct > budget else "WARNING"
            alerts.append(MonitorAlert(
                metric="scenario_worst_loss_pct",
                value=snap.scenario_worst_loss_pct,
                threshold=budget,
                severity=sev,
            ))

        # Short convexity approaching limit
        conv_limit = cfg.max_short_convexity_score
        if snap.short_convexity_score > conv_limit * cfg.monitor_convexity_warning_pct:
            alerts.append(MonitorAlert(
                metric="short_convexity_score",
                value=snap.short_convexity_score,
                threshold=conv_limit,
            ))

        # Forecast health below threshold
        if snap.forecast_health < cfg.monitor_health_warning:
            alerts.append(MonitorAlert(
                metric="forecast_health",
                value=snap.forecast_health,
                threshold=cfg.monitor_health_warning,
            ))

        # Allocation exceeds max unexpectedly
        if snap.w_vrp > cfg.w_max_vrp + 0.001:
            alerts.append(MonitorAlert(
                metric="w_vrp",
                value=snap.w_vrp,
                threshold=cfg.w_max_vrp,
                severity="CRITICAL",
            ))

        return alerts

    @property
    def history(self) -> List[DailySnapshot]:
        return list(self._history)

    def save(self, root: Path, run_id: str = "default") -> Path:
        """Persist history to CSV + JSON."""
        out_dir = root / "vrp_monitoring" / run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # JSON
        records = [s.to_dict() for s in self._history]
        json_path = out_dir / "snapshots.json"
        json_path.write_text(json.dumps(records, indent=2, default=str), encoding="utf-8")

        # CSV (flat — alerts excluded)
        rows = []
        for s in self._history:
            d = asdict(s)
            d.pop("alerts", None)
            d["alert_count"] = len(s.alerts)
            rows.append(d)
        if rows:
            df = pd.DataFrame(rows)
            df.to_csv(out_dir / "daily_telemetry.csv", index=False)

        return json_path
