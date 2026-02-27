"""
VRP daily decision audit artifact — persists every decision factor
for a single trading day, including regime, gating, sizing, scenarios,
exits, and forecast inputs.  Written even on no-trade days.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class VRPDailyAudit:
    """Full decision trace for one trading day."""

    as_of_date: str                                          # ISO format
    regime: str                                              # normal_carry | caution | crash_risk
    regime_score_components: Dict[str, float] = field(default_factory=dict)
    forecast_inputs: Optional[Dict[str, Any]] = None         # rv10_pred quantiles, health score
    gating_decisions: List[Dict[str, Any]] = field(default_factory=list)
    size_scaler_components: Dict[str, float] = field(default_factory=dict)
    scenario_estimates: Dict[str, Any] = field(default_factory=dict)
    concentration_checks: Dict[str, Any] = field(default_factory=dict)
    margin_checks: Dict[str, Any] = field(default_factory=dict)
    exit_actions: List[Dict[str, Any]] = field(default_factory=list)
    danger_zone_results: List[Dict[str, Any]] = field(default_factory=list)
    constraints_before: Dict[str, Any] = field(default_factory=dict)
    constraints_after: Dict[str, Any] = field(default_factory=dict)
    migration_warnings: List[str] = field(default_factory=list)
    allocator_decision: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, default=str)

    def save(self, root: Path) -> Path:
        """Persist audit to {root}/vrp_audits/date=YYYY-MM-DD/audit.json."""
        out_dir = root / "vrp_audits" / f"date={self.as_of_date}"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / "audit.json"
        path.write_text(self.to_json(), encoding="utf-8")
        return path

    @classmethod
    def load(cls, root: Path, as_of_date: str) -> "VRPDailyAudit":
        path = root / "vrp_audits" / f"date={as_of_date}" / "audit.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)
