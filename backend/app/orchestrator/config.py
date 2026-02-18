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
