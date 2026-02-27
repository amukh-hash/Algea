from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .broker import BrokerAdapter, PaperBrokerStub
from .calendar import MarketCalendar, Session
from .clock import normalize_asof_date, now_et
from .config import OrchestratorConfig
from .job_defs import Job, default_jobs, filtered_jobs
from .locks import LockManager
from .runner import JobRunner
from .state_store import StateStore
from .control_state import control_state
from .tick_context import TickContext
from .model_versions import record_model_versions

logger = logging.getLogger(__name__)


@dataclass
class TickResult:
    run_id: str
    asof_date: str
    session: str
    ran_jobs: list[str]
    skipped_jobs: list[str]
    failed_jobs: list[str]


class Orchestrator:
    def __init__(
        self,
        config: OrchestratorConfig | None = None,
        jobs: list[Job] | None = None,
        broker: BrokerAdapter | None = None,
        runner: JobRunner | None = None,
        telemetry: bool = False,
    ) -> None:
        self.config = config or OrchestratorConfig()
        self.calendar = MarketCalendar(self.config)
        self.state = StateStore(self.config.db_path)
        self.jobs = jobs or default_jobs()
        self.broker = broker or PaperBrokerStub()
        self.runner = runner or JobRunner()
        self.locks = LockManager(self.config.db_path)
        self._telemetry_bridge: Any = None
        if telemetry:
            try:
                from .telemetry_bridge import TelemetryBridge
                self._telemetry_bridge = TelemetryBridge()
                logger.info("Telemetry bridge enabled")
            except Exception:
                logger.warning("Telemetry bridge unavailable", exc_info=True)

    def _day_root(self, asof_date: date) -> Path:
        root = self.config.artifact_root / asof_date.isoformat()
        (root / "runs").mkdir(parents=True, exist_ok=True)
        (root / "jobs").mkdir(parents=True, exist_ok=True)
        (root / "orders").mkdir(parents=True, exist_ok=True)
        return root

    def _write_heartbeat(self, day_root: Path, now: datetime, session: Session, state: str) -> None:
        payload = {
            "timestamp": now.isoformat(),
            "session": session.value,
            "state": state,
            "mode": self.config.mode,
        }
        (day_root / "heartbeat.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def run_once(self, asof: date | datetime | None = None, forced_session: Session | None = None, dry_run: bool = False) -> TickResult:
        now = now_et()
        tick_context = TickContext()
        control_snapshot = control_state.snapshot()
        asof_date = normalize_asof_date(asof)
        day_root = self._day_root(asof_date)
        session = forced_session or self.calendar.current_session(now)
        self._write_heartbeat(day_root, now, session, "tick_start")

        if bool(control_snapshot.get("paused", False)):
            return TickResult("paused", asof_date.isoformat(), session.value, [], [], [])

        if not forced_session and not self.calendar.is_trading_day(now) and session != Session.OVERNIGHT:
            return TickResult("no_trading_day", asof_date.isoformat(), session.value, [], [], [])

        run_id = str(uuid.uuid4())
        self.state.create_run_record(run_id, asof_date.isoformat(), session.value, {"dry_run": dry_run})

        ran: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []
        failed_set: set[str] = set()

        jobs = filtered_jobs(
            self.jobs,
            session=session,
            mode=self.config.mode,
            enabled=self.config.enabled_jobs,
            disabled=self.config.disabled_jobs,
        )
        self.runner.dry_run = dry_run

        for job in jobs:
            # --- idempotency check (top priority, preserves DB status) ---
            prior = self.state.get_job_status(asof_date.isoformat(), session.value, job.name)
            if prior == "success":
                skipped.append(job.name)
                # Intentionally DO NOT call mark_job_skipped here to preserve "success" status
                continue

            # --- cooldown / min-interval check (before dependency resolution) ---
            if job.min_interval_s > 0:
                last_ok = self.state.get_last_success_at(job.name)
                if last_ok:
                    try:
                        last_ok_dt = datetime.fromisoformat(last_ok)
                    except ValueError:
                        last_ok_dt = None
                    if last_ok_dt is not None:
                        elapsed = (now.astimezone(timezone.utc) - last_ok_dt.astimezone(timezone.utc)).total_seconds()
                        if elapsed < job.min_interval_s:
                            skipped.append(job.name)
                            self.state.mark_job_skipped(run_id, asof_date.isoformat(), session.value, job.name, "skipped_interval")
                            continue

            if any(dep in failed_set for dep in job.deps):
                skipped.append(job.name)
                self.state.mark_job_skipped(run_id, asof_date.isoformat(), session.value, job.name, "blocked_by_dependency")
                continue

            self.state.mark_job_running(run_id, asof_date.isoformat(), session.value, job.name)

            with self.locks.acquire_job_lock(job.name):
                result = self.runner.run(
                    job,
                    {
                        "asof_date": asof_date.isoformat(),
                        "session": session.value,
                        "artifact_root": str(day_root),
                        "mode": self.config.mode,
                        "tick_id": run_id,
                        "dry_run": dry_run,
                        "broker": self.broker,
                        "config": self.config,
                        "control_snapshot": control_snapshot,
                        "control_snapshot_id": control_snapshot.get("snapshot_id"),
                        "tick_context": tick_context,
                    },
                    day_root / "jobs",
                )

            if result.status == "success":
                ran.append(job.name)
                self.state.mark_job_success(
                    run_id,
                    asof_date.isoformat(),
                    session.value,
                    job.name,
                    result.stdout_path,
                    result.stderr_path,
                    result.artifacts,
                )
            else:
                failed_set.add(job.name)
                failed.append(job.name)
                self.state.mark_job_failed(
                    run_id,
                    asof_date.isoformat(),
                    session.value,
                    job.name,
                    result.error_summary or "failed",
                    result.exit_code,
                    result.stdout_path,
                    result.stderr_path,
                )

        status = "failed" if failed else "success"
        tick = TickResult(run_id, asof_date.isoformat(), session.value, ran, skipped, failed)
        self.state.update_run_record(run_id, status, asdict(tick))
        (day_root / "runs" / f"{run_id}.json").write_text(json.dumps(asdict(tick), indent=2), encoding="utf-8")
        record_model_versions(
            day_root / "model_versions.json",
            tick_context.model_versions,
            run_id=run_id,
            asof_date=asof_date.isoformat(),
            session=session.value,
        )
        self._write_heartbeat(day_root, now_et(), session, status)
        logger.info("orchestrator tick complete: %s", tick)
        if self._telemetry_bridge:
            self._telemetry_bridge.emit_tick(tick, dry_run=dry_run)
        return tick
