from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime
from pathlib import Path

import pytest

from backend.app.orchestrator.broker import PaperBrokerStub
from backend.app.orchestrator.calendar import MarketCalendar, Session
from backend.app.orchestrator.config import OrchestratorConfig
from backend.app.orchestrator.job_defs import Job, handle_order_build_and_route, topo_sort
from backend.app.orchestrator.migrations import apply_migrations
from backend.app.orchestrator.orchestrator import Orchestrator


class SpyBroker(PaperBrokerStub):
    def __init__(self, account_id: str = "DU111111", is_paper: bool = True) -> None:
        super().__init__(account_id=account_id, is_paper=is_paper)
        self.place_orders_calls = 0

    def place_orders(self, orders: dict) -> dict:
        self.place_orders_calls += 1
        return super().place_orders(orders)



def _cfg(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        artifact_root=tmp_path / "artifacts",
        db_path=tmp_path / "orchestrator.sqlite3",
        mode="paper",
        paper_only=True,
    )


def test_calendar_trading_day_holiday(tmp_path):
    cal = MarketCalendar(_cfg(tmp_path))
    assert not cal.is_trading_day(datetime(2026, 1, 1, 12, 0, tzinfo=cal.tz))
    assert cal.is_trading_day(datetime(2026, 1, 2, 12, 0, tzinfo=cal.tz))


def test_current_session_boundaries(tmp_path):
    cal = MarketCalendar(_cfg(tmp_path))
    now = datetime(2026, 2, 17, 9, 30, tzinfo=cal.tz)
    assert cal.current_session(now) == Session.OPEN


def test_idempotency_runs_once_per_session(tmp_path):
    orch = Orchestrator(config=_cfg(tmp_path))
    asof = date(2026, 2, 17)
    first = orch.run_once(asof=asof, forced_session=Session.PREMARKET, dry_run=True)
    second = orch.run_once(asof=asof, forced_session=Session.PREMARKET, dry_run=True)
    assert first.ran_jobs
    assert second.ran_jobs == []
    assert second.skipped_jobs


def test_dependency_ordering_and_skip_on_failure(tmp_path):
    def fail(_: dict) -> dict:
        raise RuntimeError("risk failed")

    jobs = [
        Job("risk_checks_global", {Session.OPEN}, [], {"paper"}, 10, 0, fail),
        Job("order_build_and_route", {Session.OPEN}, ["risk_checks_global"], {"paper"}, 10, 0, lambda _: {"ok": True}),
    ]
    assert [j.name for j in topo_sort(jobs)] == ["risk_checks_global", "order_build_and_route"]

    orch = Orchestrator(config=_cfg(tmp_path), jobs=jobs)
    res = orch.run_once(asof=date(2026, 2, 17), forced_session=Session.OPEN, dry_run=False)
    assert "risk_checks_global" in res.failed_jobs
    assert "order_build_and_route" in res.skipped_jobs


def test_paper_guard_blocks_non_paper(tmp_path):
    broker = SpyBroker(account_id="U123456", is_paper=True)
    orch = Orchestrator(config=_cfg(tmp_path), broker=broker)
    orch.run_once(asof=date(2026, 2, 17), forced_session=Session.PREMARKET, dry_run=True)
    res = orch.run_once(asof=date(2026, 2, 17), forced_session=Session.OPEN, dry_run=False)
    assert "order_build_and_route" in res.failed_jobs


def test_canonical_risk_report_schema(tmp_path):
    orch = Orchestrator(config=_cfg(tmp_path), broker=SpyBroker())
    asof = date(2026, 2, 17)
    orch.run_once(asof=asof, forced_session=Session.PREMARKET, dry_run=True)
    risk_path = tmp_path / "artifacts" / asof.isoformat() / "reports" / "risk_checks.json"
    risk = json.loads(risk_path.read_text(encoding="utf-8"))
    for key in ["status", "checked_at", "asof_date", "session", "inputs", "metrics", "limits", "violations"]:
        assert key in risk
    assert "target_paths" in risk["inputs"]
    assert "per_sleeve" in risk["metrics"]


def test_open_dry_run_writes_orders_without_routing(tmp_path):
    broker = SpyBroker()
    orch = Orchestrator(config=_cfg(tmp_path), broker=broker)
    asof = date(2026, 2, 17)
    orch.run_once(asof=asof, forced_session=Session.PREMARKET, dry_run=True)
    res = orch.run_once(asof=asof, forced_session=Session.OPEN, dry_run=True)
    assert "order_build_and_route" in res.ran_jobs
    orders_path = tmp_path / "artifacts" / asof.isoformat() / "orders" / "orders.json"
    payload = json.loads(orders_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert isinstance(payload["orders"], list)
    assert "inputs" in payload
    assert "summary" in payload
    assert broker.place_orders_calls == 0


def test_order_routing_rejects_oversized_notional(tmp_path):
    root = tmp_path / "artifacts" / "2026-02-17"
    (root / "targets").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)

    for sleeve in ["core", "vrp", "selector"]:
        (root / "targets" / f"{sleeve}_targets.json").write_text(
            json.dumps({"targets": [{"symbol": "SPY", "target_weight": 1.0}]}, indent=2),
            encoding="utf-8",
        )
    (root / "reports" / "risk_checks.json").write_text(
        json.dumps(
            {
                "status": "ok",
                "checked_at": "2026-02-17T13:00:00+00:00",
                "asof_date": "2026-02-17",
                "session": "open",
                "inputs": {"target_paths": {}},
                "metrics": {"nan_or_inf": False, "gross_exposure": 1.0, "net_exposure": 0.0, "num_symbols": 1, "per_sleeve": {}},
                "limits": {},
                "violations": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="order sanity gate failed"):
        handle_order_build_and_route(
            {
                "asof_date": "2026-02-17",
                "session": "open",
                "artifact_root": str(root),
                "mode": "paper",
                "dry_run": False,
                "broker": SpyBroker(),
                "config": {
                    "account_equity": 10_000_000,
                    "max_order_notional": 1_000,
                    "max_total_order_notional": 2_000,
                    "max_orders": 50,
                    "order_notional_rounding": 10,
                },
            }
        )
    rejected = json.loads((root / "orders" / "rejected.json").read_text(encoding="utf-8"))
    assert rejected["status"] == "failed"
    assert "max_order_notional_exceeded" in rejected["reasons"] or "max_total_order_notional_exceeded" in rejected["reasons"]


def test_schema_version_newer_than_code_fails(tmp_path):
    db = tmp_path / "state.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE schema_version(version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version(version) VALUES (999)")
        conn.commit()
        with pytest.raises(RuntimeError, match="newer than supported"):
            apply_migrations(conn)
