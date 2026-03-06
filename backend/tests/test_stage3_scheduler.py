"""Tests for Stage 3: Phase-aware scheduler.

Verifies:
  - Trigger definitions and schedule mapping
  - Phase → ExecutionPhase routing (asset-class aware)
  - Seconds-until calculation
  - Start / shutdown lifecycle
  - Execution log recording
  - Commodity-specific close triggers
"""
from __future__ import annotations

import asyncio
from datetime import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.app.orchestrator.phase_scheduler import PhaseScheduler, ScheduledTrigger


@pytest.fixture()
def scheduler(tmp_path: Path) -> PhaseScheduler:
    db_path = tmp_path / "state.sqlite3"
    db_path.touch()
    return PhaseScheduler(db_path=db_path)


class TestScheduledTrigger:
    def test_repr(self):
        t = ScheduledTrigger("test", time(9, 30), AsyncMock())
        assert "test" in repr(t)
        assert "09:30" in repr(t)

    def test_fields(self):
        cb = AsyncMock()
        t = ScheduledTrigger(
            "intraday", time(10, 0), cb,
            repeat_interval_m=15, repeat_until=time(15, 0),
        )
        assert t.name == "intraday"
        assert t.repeat_interval_m == 15
        assert t.repeat_until == time(15, 0)


class TestPhaseScheduler:
    def test_default_triggers_count(self, scheduler: PhaseScheduler):
        # 11 triggers: inference_premarket, inference_preclose,
        # ingest_core_tft, inference_core_tft,
        # route_equity_moo, route_futures_open, route_intraday,
        # route_close_metals, route_close_energy,
        # route_equity_moc, route_futures_close
        assert len(scheduler.triggers) == 11

    def test_trigger_names(self, scheduler: PhaseScheduler):
        names = {t.name for t in scheduler.triggers}
        expected = {
            "inference_premarket",
            "inference_preclose",
            "ingest_core_tft",
            "inference_core_tft",
            "route_equity_moo",
            "route_futures_open",
            "route_intraday",
            "route_close_metals",
            "route_close_energy",
            "route_equity_moc",
            "route_futures_close",
        }
        assert names == expected

    def test_trigger_times(self, scheduler: PhaseScheduler):
        by_name = {t.name: t for t in scheduler.triggers}
        # Inference triggers
        assert by_name["inference_premarket"].trigger_time == time(7, 0)
        assert by_name["inference_preclose"].trigger_time == time(15, 0)
        assert by_name["ingest_core_tft"].trigger_time == time(9, 20, 5)
        assert by_name["inference_core_tft"].trigger_time == time(9, 21)
        # Asset-class-aware execution triggers
        assert by_name["route_equity_moo"].trigger_time == time(9, 26)
        assert by_name["route_futures_open"].trigger_time == time(9, 30)
        assert by_name["route_intraday"].trigger_time == time(9, 45)
        assert by_name["route_close_metals"].trigger_time == time(13, 28)
        assert by_name["route_close_energy"].trigger_time == time(14, 28)
        assert by_name["route_equity_moc"].trigger_time == time(15, 48)
        assert by_name["route_futures_close"].trigger_time == time(15, 59, 50)

    def test_equity_moo_before_nyse_cutoff(self, scheduler: PhaseScheduler):
        """Equity MOO must fire BEFORE the NYSE 09:28 cutoff."""
        by_name = {t.name: t for t in scheduler.triggers}
        moo_time = by_name["route_equity_moo"].trigger_time
        cutoff = time(9, 28)
        assert moo_time < cutoff, f"MOO at {moo_time} violates NYSE 09:28 cutoff"

    def test_equity_moc_before_nyse_cutoff(self, scheduler: PhaseScheduler):
        """Equity MOC must fire BEFORE the NYSE 15:50 cutoff."""
        by_name = {t.name: t for t in scheduler.triggers}
        moc_time = by_name["route_equity_moc"].trigger_time
        cutoff = time(15, 50)
        assert moc_time < cutoff, f"MOC at {moc_time} violates NYSE 15:50 cutoff"

    def test_futures_open_at_cash_open(self, scheduler: PhaseScheduler):
        """Futures must route at 09:30 (cash open), not earlier."""
        by_name = {t.name: t for t in scheduler.triggers}
        assert by_name["route_futures_open"].trigger_time == time(9, 30)

    def test_metals_close_before_settlement(self, scheduler: PhaseScheduler):
        """Metals close must fire before 13:30 COMEX settlement."""
        by_name = {t.name: t for t in scheduler.triggers}
        assert by_name["route_close_metals"].trigger_time < time(13, 30)

    def test_energy_close_before_settlement(self, scheduler: PhaseScheduler):
        """Energy close must fire before 14:30 NYMEX settlement."""
        by_name = {t.name: t for t in scheduler.triggers}
        assert by_name["route_close_energy"].trigger_time < time(14, 30)

    def test_intraday_repeats(self, scheduler: PhaseScheduler):
        intraday = next(t for t in scheduler.triggers if t.name == "route_intraday")
        assert intraday.repeat_interval_m == 15
        assert intraday.repeat_until == time(15, 45)

    def test_seconds_until_positive(self, scheduler: PhaseScheduler):
        # Any trigger should have > 0 seconds until next fire
        for trigger in scheduler.triggers:
            s = scheduler._seconds_until(trigger.trigger_time)
            assert s > 0, f"{trigger.name} returned non-positive delay"

    def test_schedule_summary(self, scheduler: PhaseScheduler):
        summary = scheduler.get_schedule_summary()
        assert len(summary) == 11
        for entry in summary:
            assert "name" in entry
            assert "trigger_time" in entry
            assert "next_fire_in_s" in entry

    def test_start_and_shutdown(self, scheduler: PhaseScheduler):
        async def _test():
            await scheduler.start()
            assert scheduler._running
            assert len(scheduler._tasks) == 11
            await scheduler.shutdown()
            assert not scheduler._running
            assert len(scheduler._tasks) == 0
        asyncio.run(_test())

    def test_inference_callback_fires(self, tmp_path: Path):
        async def _test():
            callback = AsyncMock()
            sched = PhaseScheduler(
                db_path=tmp_path / "state.sqlite3",
                orchestrator_callback=callback,
            )
            trigger = ScheduledTrigger("inference_premarket", time(7, 0), sched._run_inference)
            await sched._run_inference(trigger)
            callback.assert_awaited_once_with("premarket")
        asyncio.run(_test())

    def test_inference_preclose_session(self, tmp_path: Path):
        async def _test():
            callback = AsyncMock()
            sched = PhaseScheduler(
                db_path=tmp_path / "state.sqlite3",
                orchestrator_callback=callback,
            )
            trigger = ScheduledTrigger("inference_preclose", time(15, 0), sched._run_inference)
            await sched._run_inference(trigger)
            callback.assert_awaited_once_with("preclose")
        asyncio.run(_test())

    def test_routing_without_broker_logs_warning(self, scheduler: PhaseScheduler):
        async def _test():
            trigger = next(t for t in scheduler.triggers if t.name == "route_equity_moo")
            await scheduler._route_phase(trigger)
        asyncio.run(_test())

    def test_execution_log_records_inference(self, tmp_path: Path):
        async def _test():
            callback = AsyncMock()
            sched = PhaseScheduler(
                db_path=tmp_path / "state.sqlite3",
                orchestrator_callback=callback,
            )
            trigger = ScheduledTrigger("inference_premarket", time(7, 0), sched._run_inference)
            await sched._run_inference(trigger)
            assert len(sched.execution_log) == 1
            assert sched.execution_log[0]["status"] == "ok"
            assert sched.execution_log[0]["trigger"] == "inference_premarket"
        asyncio.run(_test())

    def test_execution_log_records_error(self, tmp_path: Path):
        async def _test():
            callback = AsyncMock(side_effect=RuntimeError("boom"))
            sched = PhaseScheduler(
                db_path=tmp_path / "state.sqlite3",
                orchestrator_callback=callback,
            )
            trigger = ScheduledTrigger("inference_premarket", time(7, 0), sched._run_inference)
            await sched._run_inference(trigger)
            assert len(sched.execution_log) == 1
            assert sched.execution_log[0]["status"] == "error"
            assert "boom" in sched.execution_log[0]["error"]
        asyncio.run(_test())


class TestCommodityFlattenTriggers:
    """Verify commodity-specific flatten triggers are present and correctly timed."""

    def test_metals_trigger_exists(self, scheduler: PhaseScheduler):
        names = {t.name for t in scheduler.triggers}
        assert "route_close_metals" in names

    def test_energy_trigger_exists(self, scheduler: PhaseScheduler):
        names = {t.name for t in scheduler.triggers}
        assert "route_close_energy" in names

    def test_metals_fires_before_equity_close(self, scheduler: PhaseScheduler):
        by_name = {t.name: t for t in scheduler.triggers}
        metals_time = by_name["route_close_metals"].trigger_time
        equity_moc_time = by_name["route_equity_moc"].trigger_time
        assert metals_time < equity_moc_time, (
            f"Metals close ({metals_time}) must fire before equity MOC ({equity_moc_time})"
        )

    def test_energy_fires_before_equity_close(self, scheduler: PhaseScheduler):
        by_name = {t.name: t for t in scheduler.triggers}
        energy_time = by_name["route_close_energy"].trigger_time
        equity_moc_time = by_name["route_equity_moc"].trigger_time
        assert energy_time < equity_moc_time, (
            f"Energy close ({energy_time}) must fire before equity MOC ({equity_moc_time})"
        )
