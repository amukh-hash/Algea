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
from .control_state_provider import get_control_state_provider
from .tick_context import TickContext
from .model_versions import record_model_versions
from .intent_aggregator import collect_and_validate_intents
from .snapshot_providers import (
    BrokerMarketDataProvider,
    BrokerPortfolioStateProvider,
    freeze_control_snapshot,
)
from backend.app.contracts.providers import MarketDataSnapshot, PortfolioStateSnapshot
from backend.app.core.runtime_mode import normalize_mode_alias
from backend.app.version import APP_DISPLAY, with_app_metadata

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

        # ── State DB Isolation ─────────────────────────────────────────
        # Stub/noop mode writes to a separate state DB so simulated runs
        # never poison the idempotency guards of paper/live sessions.
        if self.config.mode in ("noop", "stub"):
            stub_db = self.config.db_path.parent / "state_stub.sqlite3"
            self.config = OrchestratorConfig(
                **{
                    k: v for k, v in self.config.__dict__.items() if k != "db_path"
                },
                db_path=stub_db,
            )
            logger.info("Stub mode detected — routing state to %s", stub_db)

        self.calendar = MarketCalendar(self.config)
        self.state = StateStore(self.config.db_path)
        self.control_provider = get_control_state_provider(self.config.db_path)
        self.jobs = jobs or default_jobs()
        self.broker = broker or PaperBrokerStub()
        self.runner = runner or JobRunner()
        self.locks = LockManager(self.config.db_path)
        # ── Canonical snapshot providers (Phase 2) ────────────────────
        self._market_data_provider = BrokerMarketDataProvider(self.broker)
        self._portfolio_state_provider = BrokerPortfolioStateProvider(self.broker)
        self._telemetry_bridge: Any = None
        if telemetry:
            try:
                from .telemetry_bridge import TelemetryBridge
                self._telemetry_bridge = TelemetryBridge()
                logger.info("Telemetry bridge enabled")
            except Exception:
                logger.warning("Telemetry bridge unavailable", exc_info=True)
        logger.info("%s orchestrator initialized", APP_DISPLAY)

    def _day_root(self, asof_date: date) -> Path:
        root = self.config.artifact_root / asof_date.isoformat()
        (root / "runs").mkdir(parents=True, exist_ok=True)
        (root / "jobs").mkdir(parents=True, exist_ok=True)
        (root / "orders").mkdir(parents=True, exist_ok=True)
        return root

    def _write_heartbeat(
        self,
        day_root: Path,
        now: datetime,
        session: Session,
        state: str,
        *,
        run_id: str = "",
        control_snapshot_id: str = "",
        market_snapshot_id: str = "",
        portfolio_snapshot_id: str = "",
    ) -> None:
        payload = {
            "timestamp": now.isoformat(),
            "session": session.value,
            "state": state,
            "mode": self.config.mode,
            "run_id": run_id,
            "control_snapshot_id": control_snapshot_id,
            "market_snapshot_id": market_snapshot_id,
            "portfolio_snapshot_id": portfolio_snapshot_id,
        }
        (day_root / "heartbeat.json").write_text(json.dumps(with_app_metadata(payload), indent=2), encoding="utf-8")

    def run_once(self, asof: date | datetime | None = None, forced_session: Session | None = None, dry_run: bool = False) -> TickResult:
        now = now_et()
        tick_context = TickContext()
        asof_date = normalize_asof_date(asof)
        day_root = self._day_root(asof_date)
        session = forced_session or self.calendar.current_session(now)
        run_id = str(uuid.uuid4())
        control_snapshot = self.control_provider.snapshot(consumer="orchestrator", tick_id=run_id)

        # ── Freeze canonical snapshots (Phase 2) ──────────────────────
        typed_control = freeze_control_snapshot(control_snapshot, asof_date)
        typed_market: MarketDataSnapshot | None = None
        typed_portfolio: PortfolioStateSnapshot | None = None
        if self.config.FF_CANONICAL_SLEEVE_OUTPUTS:
            typed_market = self._market_data_provider.freeze_snapshot(asof_date, session.value)
            typed_portfolio = self._portfolio_state_provider.freeze_snapshot(asof_date)

        _ctrl_sid = typed_control.snapshot_id
        _mkt_sid = typed_market.snapshot_id if typed_market else ""
        _pf_sid = typed_portfolio.snapshot_id if typed_portfolio else ""

        self._write_heartbeat(
            day_root, now, session, "tick_start",
            run_id=run_id,
            control_snapshot_id=_ctrl_sid,
            market_snapshot_id=_mkt_sid,
            portfolio_snapshot_id=_pf_sid,
        )

        if bool(control_snapshot.get("paused", False)):
            return TickResult("paused", asof_date.isoformat(), session.value, [], [], [])

        if not forced_session and not self.calendar.is_trading_day(now) and session != Session.OVERNIGHT:
            return TickResult("no_trading_day", asof_date.isoformat(), session.value, [], [], [])

        self.state.create_run_record(run_id, asof_date.isoformat(), session.value, {"dry_run": dry_run})

        ran: list[str] = []
        skipped: list[str] = []
        failed: list[str] = []
        failed_set: set[str] = set()

        effective_mode, mode_alias_applied = normalize_mode_alias(self.config.mode)
        if mode_alias_applied:
            logger.warning(
                "Mode alias normalized for job filtering: raw=%s normalized=%s",
                self.config.mode,
                effective_mode,
            )

        jobs = filtered_jobs(
            self.jobs,
            session=session,
            mode=effective_mode,
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
                        "mode": effective_mode,
                        "mode_raw": self.config.mode,
                        "mode_alias_applied": mode_alias_applied,
                        "tick_id": run_id,
                        "dry_run": dry_run,
                        "broker": self.broker,
                        "config": self.config,
                        "control_snapshot": control_snapshot,
                        "control_snapshot_id": control_snapshot.get("snapshot_id"),
                        "tick_context": tick_context,
                        # ── Canonical snapshots (Phase 2) ─────────
                        "typed_control_snapshot": typed_control,
                        "typed_market_snapshot": typed_market,
                        "typed_portfolio_snapshot": typed_portfolio,
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

        # ── Intent Aggregation Barrier ────────────────────────────────
        # After all signal-generation jobs, collect *_intents.json and
        # push through the atomic risk gateway.
        try:
            agg_result = collect_and_validate_intents(
                artifact_root=day_root,
                db_path=self.config.db_path,
                asof_date=asof_date.isoformat(),
                tick_id=run_id,
            )
            if agg_result["status"] == "rejected":
                logger.error("Risk gateway REJECTED intents: %s", agg_result.get("error"))
            elif agg_result["n_collected"] > 0:
                logger.info(
                    "Intent barrier: %d intents collected, %d validated",
                    agg_result["n_collected"], agg_result["n_validated"],
                )
            if int(agg_result.get("n_target_intent_files", 0)) > 0:
                logger.warning(
                    "Non-canonical intent path active: %d *_intents.json files under targets/",
                    int(agg_result.get("n_target_intent_files", 0)),
                )
        except Exception as exc:
            logger.exception("Intent aggregation barrier failed — this may cause orders to be stale or missing")
            failed.append("intent_aggregation")
            failed_set.add("intent_aggregation")

        status = "failed" if failed else "success"
        tick = TickResult(run_id, asof_date.isoformat(), session.value, ran, skipped, failed)
        self.state.update_run_record(run_id, status, asdict(tick))
        (day_root / "runs" / f"{run_id}.json").write_text(
            json.dumps(with_app_metadata(asdict(tick)), indent=2),
            encoding="utf-8",
        )
        record_model_versions(
            day_root / "model_versions.json",
            tick_context.model_versions,
            run_id=run_id,
            asof_date=asof_date.isoformat(),
            session=session.value,
        )
        self._write_heartbeat(day_root, now_et(), session, status)
        logger.info("orchestrator tick complete: %s", tick)

        # ── ZMQ Bridge: push tick results to native frontend ──────────
        # Throttled to 1Hz max to prevent GIL saturation at high tick rates.
        try:
            import time as _time
            _now = _time.time()
            _last_bridge = getattr(self, '_last_bridge_ts', 0.0)
            if (_now - _last_bridge) >= 1.0 or failed:
                from backend.app.api.zmq_bridge import bridge_control_snapshot, bridge_control_mutation
                bridge_control_snapshot(control_snapshot)
                bridge_control_mutation("tick_complete", {
                    "run_id": run_id,
                    "session": session.value,
                    "status": status,
                    "ran_jobs": ran,
                    "failed_jobs": failed,
                })
                self._last_bridge_ts = _now
        except Exception:
            logger.debug("ZMQ bridge tick publish skipped", exc_info=True)

        if self._telemetry_bridge:
            self._telemetry_bridge.emit_tick(tick, dry_run=dry_run)
        return tick
