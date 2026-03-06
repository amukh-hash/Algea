"""Plugin protocol for trading sleeve decoupling.

Replaces the monolithic ``job_defs.py`` with a strict interface contract.
The Orchestrator core contains zero ML logic — sleeves are loaded
dynamically via ``importlib`` inside isolated GPU worker processes.

Resolves **F9** (monolithic coupling).
"""
from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class TradingSleevePlugin(Protocol):
    """Interface contract for trading sleeve plugins.

    Implementations are executed inside the isolated GPU worker process
    managed by ``GPUProcessSupervisor``.  They must:

    1. Read input data from ``context["artifact_dir"]``
    2. Run inference using models from ``model_cache`` (or load & cache)
    3. Write ``TargetIntent`` JSON to ``context["artifact_dir"]/intents.json``

    No direct broker calls are permitted.
    """

    def execute(self, context: dict[str, Any], model_cache: Dict[str, Any]) -> None:
        """Execute the sleeve's inference pipeline.

        Parameters
        ----------
        context : dict
            Contains ``artifact_dir``, ``asof_date``, ``session``, ``run_id``,
            and other orchestrator-provided metadata.
        model_cache : dict
            Persistent model cache that survives across ticks.  Plugins
            should load models here on first call and reuse on subsequent
            calls to avoid NVMe reload latency.
        """
        ...
