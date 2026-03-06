"""Phase-aware scheduler for decoupled inference + execution pipeline.

Separates the trading day into two independent pipes:

1. **Inference triggers** — run the DAG orchestrator to generate
   ``TargetIntent`` JSON from ML models (no broker calls).
2. **Execution triggers** — call ``route_phase_orders()`` at precise
   auction windows to route pending intents through the broker.

Uses ``asyncio`` scheduling by default.  If ``apscheduler`` is installed,
it can optionally be used for higher precision and persistence.

Usage
-----
::

    from backend.app.orchestrator.phase_scheduler import PhaseScheduler

    scheduler = PhaseScheduler(db_path="state.sqlite3", broker=my_broker)
    await scheduler.start()       # runs until cancelled
    await scheduler.shutdown()
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Callable, Coroutine
from zoneinfo import ZoneInfo

from backend.app.core.risk_gateway import route_phase_orders
from backend.app.core.schemas import ExecutionPhase

logger = logging.getLogger(__name__)

EASTERN_TZ = ZoneInfo("America/New_York")


def _now_et() -> datetime:
    return datetime.now(EASTERN_TZ)


class ScheduledTrigger:
    """A single scheduled trigger within the trading day."""

    def __init__(
        self,
        name: str,
        trigger_time: time,
        callback: Callable[..., Coroutine[Any, Any, Any]],
        *,
        repeat_interval_m: int | None = None,
        repeat_until: time | None = None,
    ) -> None:
        self.name = name
        self.trigger_time = trigger_time
        self.callback = callback
        self.repeat_interval_m = repeat_interval_m
        self.repeat_until = repeat_until

    def __repr__(self) -> str:
        return f"<Trigger {self.name} @ {self.trigger_time}>"


class PhaseScheduler:
    """Manages inference + execution triggers for the trading day.

    Parameters
    ----------
    db_path : str | Path
        SQLite state database for ``route_phase_orders``.
    broker
        Broker protocol instance (``get_account_equity``, ``get_positions``,
        ``get_price``, ``place_order``).
    orchestrator_callback
        Async callable to run the DAG orchestrator inference pass.
        Signature: ``async (session: str) -> None``
    """

    def __init__(
        self,
        db_path: str | Path,
        broker: Any = None,
        orchestrator_callback: Callable[..., Coroutine[Any, Any, Any]] | None = None,
        config: Any = None,
    ) -> None:
        self.db_path = str(db_path)
        self.broker = broker
        self.orchestrator_callback = orchestrator_callback
        self.config = config
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self.triggers = self._build_triggers()
        self.execution_log: list[dict[str, Any]] = []

    def _build_triggers(self) -> list[ScheduledTrigger]:
        """Define the standard US equity trading day triggers."""
        triggers = []

        # ── Inference Triggers ───────────────────────────────────────
        # T1: Pre-market inference (generate intents for the day)
        triggers.append(
            ScheduledTrigger(
                name="inference_premarket",
                trigger_time=time(7, 0),  # 07:00 EST
                callback=self._run_inference,
            )
        )
        # T2: Pre-close inference (generate close-auction intents)
        triggers.append(
            ScheduledTrigger(
                name="inference_preclose",
                trigger_time=time(15, 0),  # 15:00 EST
                callback=self._run_inference,
            )
        )

        # ── TFT Core Reversal Triggers ──────────────────────────────
        # T2a: Fetch data right after the 09:20 bar closes
        triggers.append(
            ScheduledTrigger(
                name="ingest_core_tft",
                trigger_time=time(9, 20, 5),  # 09:20:05 EST — finalized 09:15-09:20 bar
                callback=self._run_inference,
            )
        )
        # T2b: Run GPU inference immediately after data is written
        triggers.append(
            ScheduledTrigger(
                name="inference_core_tft",
                trigger_time=time(9, 21, 0),  # 8 minutes before 09:29 routing
                callback=self._run_inference,
            )
        )

        # ── Execution Triggers (Asset-Class Aware) ────────────────────
        # T3a: Equity MOO — safely beats NYSE 09:28 cutoff
        triggers.append(
            ScheduledTrigger(
                name="route_equity_moo",
                trigger_time=time(9, 26),  # 09:26:00 ET (2-min before 09:28 cutoff)
                callback=self._route_phase,
            )
        )
        # T3b: Futures MKT at exact cash open (no MOO auction on CME)
        triggers.append(
            ScheduledTrigger(
                name="route_futures_open",
                trigger_time=time(9, 30),  # 09:30:00 ET — exact cash equity open
                callback=self._route_phase,
            )
        )
        # T4: Intraday — continuous routing every 15 minutes
        triggers.append(
            ScheduledTrigger(
                name="route_intraday",
                trigger_time=time(9, 45),  # Start 09:45 ET
                callback=self._route_phase,
                repeat_interval_m=15,
                repeat_until=time(15, 45),  # Last trigger 15:45 ET
            )
        )
        # T5a: Commodity metals close — GC/SI/HG 2-min before 13:30 settlement
        triggers.append(
            ScheduledTrigger(
                name="route_close_metals",
                trigger_time=time(13, 28),  # 13:28:00 ET
                callback=self._route_phase,
            )
        )
        # T5b: Commodity energy close — CL 2-min before 14:30 settlement
        triggers.append(
            ScheduledTrigger(
                name="route_close_energy",
                trigger_time=time(14, 28),  # 14:28:00 ET
                callback=self._route_phase,
            )
        )
        # T6a: Equity MOC — safely beats NYSE 15:50 cutoff
        triggers.append(
            ScheduledTrigger(
                name="route_equity_moc",
                trigger_time=time(15, 48),  # 15:48:00 ET (2-min before 15:50 cutoff)
                callback=self._route_phase,
            )
        )
        # T6b: Futures MKT close — 10s before cash close (no CME TAS conflicts)
        triggers.append(
            ScheduledTrigger(
                name="route_futures_close",
                trigger_time=time(15, 59, 50),  # 15:59:50 ET
                callback=self._route_phase,
            )
        )

        # ── Conditional: StatArb 5-minute routing ────────────────────
        # Only registered when the StatArb sleeve is enabled (not in
        # stub/lockdown mode).  4-second offset guarantees the prior
        # 5-minute candle has closed and propagated over the broker API.
        if self.config and getattr(self.config, "enable_statarb_sleeve", False):
            triggers.append(
                ScheduledTrigger(
                    name="route_statarb",
                    trigger_time=time(9, 35, 4),  # 09:35:04 ET (4s propagation delay)
                    callback=self._route_phase,
                    repeat_interval_m=5,
                    repeat_until=time(15, 55, 4),  # 15:55:04 ET
                )
            )

        return triggers

    async def _run_inference(self, trigger: ScheduledTrigger) -> None:
        """Execute the DAG orchestrator inference pass."""
        session = "premarket" if trigger.trigger_time.hour < 12 else "preclose"
        logger.info("Inference trigger fired: %s (session=%s)", trigger.name, session)

        if self.orchestrator_callback:
            try:
                await self.orchestrator_callback(session)
                self.execution_log.append({
                    "trigger": trigger.name,
                    "time": _now_et().isoformat(),
                    "status": "ok",
                })
            except Exception as exc:
                logger.error("Inference trigger %s failed: %s", trigger.name, exc)
                self.execution_log.append({
                    "trigger": trigger.name,
                    "time": _now_et().isoformat(),
                    "status": "error",
                    "error": str(exc),
                })
        else:
            logger.warning("No orchestrator_callback configured — skipping %s", trigger.name)

    async def _route_phase(self, trigger: ScheduledTrigger) -> None:
        """Execute phase-aware order routing."""
        # Map trigger name → ExecutionPhase
        phase_map = {
            "route_equity_moo": ExecutionPhase.AUCTION_OPEN,
            "route_futures_open": ExecutionPhase.FUTURES_OPEN,
            "route_intraday": ExecutionPhase.INTRADAY,
            "route_statarb": ExecutionPhase.INTRADAY,
            "route_close_metals": ExecutionPhase.COMMODITY_CLOSE,
            "route_close_energy": ExecutionPhase.COMMODITY_CLOSE,
            "route_equity_moc": ExecutionPhase.AUCTION_CLOSE,
            "route_futures_close": ExecutionPhase.FUTURES_CLOSE,
        }
        phase = phase_map.get(trigger.name, ExecutionPhase.INTRADAY)
        asof_date = _now_et().strftime("%Y-%m-%d")

        logger.info(
            "Routing trigger fired: %s → phase=%s asof=%s",
            trigger.name, phase.value, asof_date,
        )

        if not self.broker:
            logger.warning("No broker configured — skipping %s", trigger.name)
            return

        try:
            result = route_phase_orders(
                db_path=self.db_path,
                broker=self.broker,
                phase=phase,
                asof_date=asof_date,
            )
            self.execution_log.append({
                "trigger": trigger.name,
                "phase": phase.value,
                "time": _now_et().isoformat(),
                "status": "ok",
                "result": result,
            })
            logger.info(
                "Routing %s complete: %d routed, %d skipped",
                phase.value,
                result.get("routed_count", 0),
                result.get("skipped_count", 0),
            )
        except Exception as exc:
            logger.error("Routing trigger %s failed: %s", trigger.name, exc)
            self.execution_log.append({
                "trigger": trigger.name,
                "phase": phase.value,
                "time": _now_et().isoformat(),
                "status": "error",
                "error": str(exc),
            })

    def _seconds_until(self, target: time) -> float:
        """Compute seconds from now until the next occurrence of *target* (EST)."""
        now = _now_et()
        target_dt = datetime.combine(now.date(), target, tzinfo=EASTERN_TZ)
        if target_dt <= now:
            target_dt += timedelta(days=1)
        return (target_dt - now).total_seconds()

    async def _schedule_trigger(self, trigger: ScheduledTrigger) -> None:
        """Run a single trigger, sleeping until its fire time."""
        while self._running:
            delay = self._seconds_until(trigger.trigger_time)
            logger.debug(
                "Trigger %s sleeping %.0fs until %s EST",
                trigger.name, delay, trigger.trigger_time,
            )
            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                return

            if not self._running:
                return

            await trigger.callback(trigger)

            # Handle repeating triggers (e.g. intraday every 15m)
            if trigger.repeat_interval_m and trigger.repeat_until:
                while self._running:
                    now = _now_et()
                    if now.time() >= trigger.repeat_until:
                        break
                    try:
                        await asyncio.sleep(trigger.repeat_interval_m * 60)
                    except asyncio.CancelledError:
                        return
                    if not self._running:
                        return
                    if _now_et().time() >= trigger.repeat_until:
                        break
                    await trigger.callback(trigger)

    async def start(self) -> None:
        """Start all triggers as concurrent asyncio tasks."""
        self._running = True
        logger.info(
            "PhaseScheduler starting with %d triggers", len(self.triggers),
        )
        for trigger in self.triggers:
            task = asyncio.create_task(
                self._schedule_trigger(trigger),
                name=f"phase_sched_{trigger.name}",
            )
            self._tasks.append(task)
        logger.info("PhaseScheduler active — awaiting market triggers")

    async def shutdown(self) -> None:
        """Cancel all trigger tasks."""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("PhaseScheduler shut down")

    def get_schedule_summary(self) -> list[dict[str, Any]]:
        """Return human-readable schedule for diagnostics."""
        out = []
        for t in self.triggers:
            entry: dict[str, Any] = {
                "name": t.name,
                "trigger_time": t.trigger_time.strftime("%H:%M EST"),
                "next_fire_in_s": round(self._seconds_until(t.trigger_time)),
            }
            if t.repeat_interval_m:
                entry["repeat_every_m"] = t.repeat_interval_m
                entry["repeat_until"] = t.repeat_until.strftime("%H:%M EST") if t.repeat_until else None
            out.append(entry)
        return out
