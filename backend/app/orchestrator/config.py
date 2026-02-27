from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SessionWindow:
    start: str
    end: str


@dataclass
class OrchestratorConfig:
    timezone: str = field(default_factory=lambda: os.getenv("ORCH_TIMEZONE", "America/New_York"))
    exchange: str = field(default_factory=lambda: os.getenv("ORCH_EXCHANGE", "XNYS"))
    mode: str = field(default_factory=lambda: os.getenv("ORCH_MODE", "paper"))
    poll_interval_s: int = field(default_factory=lambda: int(os.getenv("ORCH_POLL_INTERVAL_S", "60")))
    artifact_root: Path = field(
        default_factory=lambda: Path(os.getenv("ORCH_ARTIFACT_ROOT", "backend/artifacts/orchestrator"))
    )
    db_path: Path = field(
        default_factory=lambda: Path(os.getenv("ORCH_DB_PATH", "backend/artifacts/orchestrator_state/state.sqlite3"))
    )
    enabled_jobs: list[str] = field(default_factory=list)
    disabled_jobs: list[str] = field(default_factory=list)
    paper_only: bool = field(default_factory=lambda: os.getenv("ORCH_PAPER_ONLY", "1") == "1")
    account_equity: float = field(default_factory=lambda: float(os.getenv("ORCH_ACCOUNT_EQUITY", "100000")))
    max_order_notional: float = field(default_factory=lambda: float(os.getenv("ORCH_MAX_ORDER_NOTIONAL", "5000")))
    max_total_order_notional: float = field(default_factory=lambda: float(os.getenv("ORCH_MAX_TOTAL_ORDER_NOTIONAL", "25000")))
    max_orders: int = field(default_factory=lambda: int(os.getenv("ORCH_MAX_ORDERS", "20")))
    enable_chronos2_sleeve: bool = field(default_factory=lambda: os.getenv("ENABLE_CHRONOS2_SLEEVE", "0") == "1")
    enable_smoe_selector: bool = field(default_factory=lambda: os.getenv("ENABLE_SMOE_SELECTOR", "0") == "1")
    selector_model_alias: str = field(default_factory=lambda: os.getenv("SELECTOR_MODEL_ALIAS", "prod"))
    enable_vol_surface_vrp: bool = field(default_factory=lambda: os.getenv("ENABLE_VOL_SURFACE_VRP", "0") == "1")
    vrp_model_alias: str = field(default_factory=lambda: os.getenv("VRP_MODEL_ALIAS", "prod"))
    enable_statarb_sleeve: bool = field(default_factory=lambda: os.getenv("ENABLE_STATARB_SLEEVE", "0") == "1")
    itransformer_model_alias: str = field(default_factory=lambda: os.getenv("ITRANSFORMER_MODEL_ALIAS", "prod"))
    session_windows: dict[str, SessionWindow] = field(
        default_factory=lambda: {
            "PREMARKET": SessionWindow("07:00", "09:25"),
            "OPEN": SessionWindow("09:25", "09:35"),
            "INTRADAY": SessionWindow("09:35", "15:45"),
            "PRECLOSE": SessionWindow("15:45", "15:58"),
            "CLOSE": SessionWindow("15:58", "16:10"),
            "AFTERHOURS": SessionWindow("16:10", "20:00"),
        }
    )
