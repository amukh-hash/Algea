from __future__ import annotations

from .calendar import Session
from .config import OrchestratorConfig

__all__ = ["Session", "OrchestratorConfig", "Orchestrator"]


def __getattr__(name: str):
    if name == "Orchestrator":
        from .orchestrator import Orchestrator

        return Orchestrator
    raise AttributeError(name)
