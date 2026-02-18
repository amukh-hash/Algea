"""Bridge orchestrator ticks → telemetry system.

After each ``Orchestrator.run_once`` tick, call ``emit_tick`` to create/update
a telemetry Run so orchestrator activity appears on the Execution dashboard.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from backend.app.orchestrator.orchestrator import TickResult

from backend.app.telemetry.emitter import TelemetryEmitter
from backend.app.telemetry.schemas import EventLevel, EventType, RunStatus, RunType
from backend.app.telemetry.storage import TelemetryStorage

logger = logging.getLogger(__name__)


class TelemetryBridge:
    """Emit orchestrator events into the telemetry system."""

    def __init__(self, storage: TelemetryStorage | None = None) -> None:
        self.storage = storage or TelemetryStorage()
        self.emitter = TelemetryEmitter(self.storage)

    def emit_tick(self, tick: TickResult, dry_run: bool = False) -> str | None:
        """Publish a single orchestrator tick as a telemetry run.

        Returns the telemetry run_id, or None on error.
        """
        try:
            return self._emit(tick, dry_run)
        except Exception:
            logger.exception("telemetry bridge failed for tick %s", tick.run_id)
            return None

    def _emit(self, tick: TickResult, dry_run: bool) -> str:
        now = datetime.now(timezone.utc)

        # Determine run status
        if tick.failed_jobs:
            run_status = RunStatus.error
        elif tick.ran_jobs:
            run_status = RunStatus.completed
        else:
            run_status = RunStatus.completed  # all skipped is still "completed"

        # Start a run
        run_id = self.emitter.start_run(
            run_type=RunType.sleeve_paper,
            name=f"Orchestrator · {tick.session} · {tick.asof_date}",
            sleeve_name="orchestrator",
            tags=["orchestrator", tick.session, "paper" if not dry_run else "dry-run"],
            meta={
                "asof_date": tick.asof_date,
                "session": tick.session,
                "dry_run": dry_run,
                "orch_run_id": tick.run_id,
                "ran_jobs": tick.ran_jobs,
                "failed_jobs": tick.failed_jobs,
                "skipped_jobs": tick.skipped_jobs,
            },
        )

        # Emit per-job events
        for job_name in tick.ran_jobs:
            self.emitter.emit_event(
                run_id,
                EventLevel.info,
                EventType.DECISION_MADE,
                f"Job {job_name} completed successfully",
                ts=now,
                payload={"job": job_name, "status": "success"},
            )

        for job_name in tick.failed_jobs:
            self.emitter.emit_event(
                run_id,
                EventLevel.error,
                EventType.RISK_LIMIT,
                f"Job {job_name} failed",
                ts=now,
                payload={"job": job_name, "status": "failed"},
            )

        for job_name in tick.skipped_jobs:
            self.emitter.emit_event(
                run_id,
                EventLevel.info,
                EventType.CHECKPOINT_SAVED,
                f"Job {job_name} skipped",
                ts=now,
                payload={"job": job_name, "status": "skipped"},
            )

        # Emit summary metrics
        self.emitter.emit_metric(run_id, "jobs_ran", float(len(tick.ran_jobs)), ts=now)
        self.emitter.emit_metric(run_id, "jobs_failed", float(len(tick.failed_jobs)), ts=now)
        self.emitter.emit_metric(run_id, "jobs_skipped", float(len(tick.skipped_jobs)), ts=now)

        # Finalize
        self.emitter.set_status(run_id, run_status)
        return run_id
