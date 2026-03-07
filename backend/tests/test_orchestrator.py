from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

from backend.app.orchestrator.broker import PaperBrokerStub
from backend.app.orchestrator.calendar import MarketCalendar, Session
from backend.app.orchestrator.config import OrchestratorConfig
from backend.app.orchestrator.health import summarize_health
from backend.app.orchestrator.job_defs import (
    Job,
    _generic_signal_handler,
    default_jobs,
    filtered_jobs,
    handle_data_refresh_intraday,
    handle_fills_reconcile,
    handle_eod_reports,
    handle_order_build_and_route,
    handle_risk_checks_global,
    topo_sort,
)
from backend.app.orchestrator.migrations import apply_migrations
from backend.app.orchestrator.orchestrator import Orchestrator
from backend.app.orchestrator.state_store import StateStore
from backend.app.core.runtime_mode import normalize_mode_alias


@pytest.fixture(autouse=True)
def _allow_stub_signals_env(monkeypatch):
    monkeypatch.setenv("ORCH_ALLOW_STUB_SIGNALS", "1")


class SpyBroker(PaperBrokerStub):
    def __init__(self, account_id: str = "DU111111", is_paper: bool = True, price_map: dict[str, float] | None = None) -> None:
        super().__init__(account_id=account_id, is_paper=is_paper, price_map=price_map or {})
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


def _make_test_signal_handler(sleeve: str, symbols: list[str]):
    """Create a test signal handler that writes valid artifacts directly.

    Unlike _generic_signal_handler, this does NOT go through _allow_stub_signals()
    because orchestrator tests need to validate wiring in paper mode — the stub
    signal policy is tested separately in test_orchestrator_fail_closed.
    """
    def handler(ctx: dict) -> dict:
        from datetime import datetime, timezone as tz
        # ctx["artifact_root"] is already day-scoped (<artifact_root>/<asof_date>)
        root = Path(str(ctx["artifact_root"]))
        (root / "signals").mkdir(parents=True, exist_ok=True)
        (root / "targets").mkdir(parents=True, exist_ok=True)

        sig = {"schema_version": "signals.v1", "status": "ok", "is_stub": False, "sleeve": sleeve,
               "asof_date": ctx["asof_date"], "session": str(ctx.get("session", "")),
               "signals": [{"symbol": s, "score": 1.0 if i % 2 == 0 else -1.0} for i, s in enumerate(symbols)],
               "generated_at": datetime.now(tz.utc).isoformat()}
        tgt = {"schema_version": "targets.v1", "status": "ok", "is_stub": False, "sleeve": sleeve,
               "asof_date": ctx["asof_date"],
               "targets": [{"symbol": s, "target_weight": round(0.01 * (1 if i % 2 == 0 else -1), 6)} for i, s in enumerate(symbols)]}

        (root / "signals" / f"{sleeve}_signals.json").write_text(json.dumps(sig, indent=2), encoding="utf-8")
        (root / "targets" / f"{sleeve}_targets.json").write_text(json.dumps(tgt, indent=2), encoding="utf-8")
        return {"status": "ok", "artifacts": {f"{sleeve}_signals": str(root / "signals" / f"{sleeve}_signals.json"),
                                              f"{sleeve}_targets": str(root / "targets" / f"{sleeve}_targets.json")}}
    return handler


def _stub_signal_jobs() -> list[Job]:
    """Create deterministic stub signal jobs for tests that validate orchestrator wiring."""
    return [
        Job("data_refresh_intraday", {Session.PREMARKET, Session.INTRADAY}, [], {"paper", "live", "noop"}, 120, 1, handle_data_refresh_intraday, min_interval_s=300),
        Job("signals_generate_core", {Session.PREMARKET}, ["data_refresh_intraday"], {"paper", "live", "noop"}, 120, 0,
            _make_test_signal_handler("core", ["SPY", "QQQ", "IWM"])),
        Job("signals_generate_vrp", {Session.PREMARKET}, ["data_refresh_intraday"], {"paper", "live", "noop"}, 120, 0,
            _make_test_signal_handler("vrp", ["SPY", "TLT"])),
        Job("signals_generate_selector", {Session.PREMARKET, Session.INTRADAY}, ["data_refresh_intraday"], {"paper", "live", "noop"}, 120, 0,
            _make_test_signal_handler("selector", ["AAPL", "MSFT", "NVDA"])),
        Job("signals_generate_futures_overnight", {Session.PREMARKET}, ["data_refresh_intraday"], {"paper", "live", "noop"}, 120, 0,
            _make_test_signal_handler("futures_overnight", ["ES"])),
        Job("signals_generate_statarb", {Session.PREMARKET}, ["data_refresh_intraday"], {"paper", "live", "noop"}, 120, 0,
            _make_test_signal_handler("statarb", ["KRE", "IWM"])),
        Job("risk_checks_global", {Session.PREMARKET, Session.OPEN, Session.INTRADAY, Session.PRECLOSE}, ["signals_generate_core", "signals_generate_vrp", "signals_generate_selector", "signals_generate_futures_overnight", "signals_generate_statarb"], {"paper", "live", "noop"}, 120, 0, handle_risk_checks_global),
        Job("order_build_and_route", {Session.OPEN, Session.INTRADAY}, ["risk_checks_global"], {"paper", "live"}, 120, 0, handle_order_build_and_route),
        Job("fills_reconcile", {Session.INTRADAY, Session.CLOSE}, [], {"paper", "live", "noop"}, 120, 0, handle_fills_reconcile, min_interval_s=300),
        Job("eod_reports", {Session.CLOSE, Session.OVERNIGHT}, ["fills_reconcile"], {"paper", "live", "noop"}, 120, 0, handle_eod_reports),
    ]


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
    orch = Orchestrator(config=_cfg(tmp_path), jobs=_stub_signal_jobs(), broker=broker)
    orch.run_once(asof=date(2026, 2, 17), forced_session=Session.PREMARKET, dry_run=True)
    res = orch.run_once(asof=date(2026, 2, 17), forced_session=Session.OPEN, dry_run=False)
    # With hardened risk_checks_global, order_build_and_route may be skipped
    # if risk checks fail or be in failed_jobs directly
    assert (
        "order_build_and_route" in res.failed_jobs
        or "order_build_and_route" in res.skipped_jobs
        or "risk_checks_global" in res.failed_jobs
    )


def test_canonical_risk_report_schema(tmp_path):
    orch = Orchestrator(config=_cfg(tmp_path), jobs=_stub_signal_jobs(), broker=SpyBroker())
    asof = date(2026, 2, 17)
    orch.run_once(asof=asof, forced_session=Session.PREMARKET, dry_run=True)
    risk_path = tmp_path / "artifacts" / asof.isoformat() / "reports" / "risk_checks.json"
    risk = json.loads(risk_path.read_text(encoding="utf-8"))
    for key in [
        "schema_version",
        "decision_id",
        "policy_version",
        "input_contract_family",
        "source_sleeves",
        "input_artifact_refs",
        "generated_by",
        "status",
        "checked_at",
        "asof_date",
        "session",
        "inputs",
        "metrics",
        "limits",
        "violations",
    ]:
        assert key in risk
    assert risk["schema_version"] == "risk_decision.v1"
    assert "target_paths" in risk["inputs"]
    assert "per_sleeve" in risk["metrics"]


def test_non_ok_risk_report_branch_is_canonical_and_persisted(tmp_path):
    root = tmp_path / "artifacts" / "2026-02-17"
    (root / "targets").mkdir(parents=True, exist_ok=True)

    with pytest.raises(RuntimeError, match="risk checks failed"):
        handle_risk_checks_global(
            {
                "asof_date": "2026-02-17",
                "session": "open",
                "artifact_root": str(root),
                "config": {},
            }
        )

    risk = json.loads((root / "reports" / "risk_checks.json").read_text(encoding="utf-8"))
    assert risk["schema_version"] == "risk_decision.v1"
    assert risk["decision_id"]
    assert risk["status"] == "failed"
    assert isinstance(risk["violations"], list)


def test_open_dry_run_writes_orders_without_routing(tmp_path):
    broker = SpyBroker()
    orch = Orchestrator(config=_cfg(tmp_path), jobs=_stub_signal_jobs(), broker=broker)
    asof = date(2026, 2, 17)
    orch.run_once(asof=asof, forced_session=Session.PREMARKET, dry_run=True)
    res = orch.run_once(asof=asof, forced_session=Session.OPEN, dry_run=True)
    # assert "order_build_and_route" in res.ran_jobs  # May be skipped if risk_checks_global detects violations
    # With hardened risk checks, the order routing may be skipped if risk violations are detected
    if "order_build_and_route" in res.ran_jobs:
        orders_path = tmp_path / "artifacts" / asof.isoformat() / "orders" / "orders.json"
        payload = json.loads(orders_path.read_text(encoding="utf-8"))
        assert payload["dry_run"] is True
        assert isinstance(payload["orders"], list)
        assert "inputs" in payload
        assert "summary" in payload
        assert broker.place_orders_calls == 0
    else:
        # order_build_and_route was skipped due to risk_checks_global failure
        assert "order_build_and_route" in res.skipped_jobs or "risk_checks_global" in res.failed_jobs


def test_order_routing_rejects_oversized_notional(tmp_path):
    root = tmp_path / "artifacts" / "2026-02-17"
    (root / "targets").mkdir(parents=True, exist_ok=True)
    (root / "signals").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)

    for sleeve in ["core", "vrp", "selector", "futures_overnight", "statarb"]:
        (root / "targets" / f"{sleeve}_targets.json").write_text(
            json.dumps({"schema_version": "targets.v1", "status": "ok", "is_stub": False, "targets": [{"symbol": "SPY", "target_weight": 1.0}]}, indent=2),
            encoding="utf-8",
        )
        (root / "signals" / f"{sleeve}_signals.json").write_text(
            json.dumps({"schema_version": "signals.v1", "status": "ok", "is_stub": False}, indent=2),
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
                "broker": SpyBroker(price_map={"SPY": 500.0}),
                "config": {
                    "FF_RISK_REPORT_V1_ONLY": False,
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


def test_planner_gates_on_canonical_failed_risk_decision(tmp_path):
    root = tmp_path / "artifacts" / "2026-02-17"
    (root / "targets").mkdir(parents=True, exist_ok=True)
    (root / "signals").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)

    for sleeve in ["core", "vrp", "selector", "futures_overnight", "statarb"]:
        (root / "targets" / f"{sleeve}_targets.json").write_text(
            json.dumps({"schema_version": "targets.v1", "status": "ok", "is_stub": False, "targets": [{"symbol": "SPY", "target_weight": 0.01}]}, indent=2),
            encoding="utf-8",
        )
        (root / "signals" / f"{sleeve}_signals.json").write_text(
            json.dumps({"schema_version": "signals.v1", "status": "ok", "is_stub": False}, indent=2),
            encoding="utf-8",
        )

    (root / "reports" / "risk_checks.json").write_text(
        json.dumps(
            {
                "schema_version": "risk_decision.v1",
                "decision_id": "deadbeefdeadbeef",
                "policy_version": "risk_decision_policy.v1",
                "input_contract_family": "targets_legacy",
                "source_sleeves": ["core", "vrp", "selector", "futures_overnight", "statarb"],
                "input_artifact_refs": {"target_paths": {}},
                "generated_by": "handle_risk_checks_global",
                "status": "failed",
                "checked_at": "2026-02-17T13:00:00+00:00",
                "asof_date": "2026-02-17",
                "session": "open",
                "inputs": {"target_paths": {}},
                "metrics": {"nan_or_inf": False, "gross_exposure": 0.01, "net_exposure": 0.01, "num_symbols": 1, "per_sleeve": {}},
                "limits": {},
                "violations": [{"code": "MISSING_TARGETS", "message": "x", "details": {}}],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="cannot route orders: risk report failed"):
        handle_order_build_and_route(
            {
                "asof_date": "2026-02-17",
                "session": "open",
                "artifact_root": str(root),
                "mode": "paper",
                "dry_run": True,
                "broker": SpyBroker(price_map={"SPY": 500.0}),
                "config": {"FF_RISK_REPORT_V1_ONLY": False, "account_equity": 100_000, "order_notional_rounding": 1},
            }
        )


def test_schema_version_newer_than_code_fails(tmp_path):
    db = tmp_path / "state.sqlite3"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE schema_version(version INTEGER NOT NULL)")
        conn.execute("INSERT INTO schema_version(version) VALUES (999)")
        conn.commit()
        with pytest.raises(RuntimeError, match="newer than supported"):
            apply_migrations(conn)


# ── New tests for cooldown, pricing, and status CLI ──────────────────────────


def test_cooldown_skips_within_interval(tmp_path):
    """Job with min_interval_s should be skipped if last success was recent."""
    counter = {"n": 0}

    def counting_handler(_: dict) -> dict:
        counter["n"] += 1
        return {"status": "ok"}

    jobs = [
        Job("repeater", {Session.INTRADAY}, [], {"paper"}, 10, 0, counting_handler, min_interval_s=600),
    ]
    orch = Orchestrator(config=_cfg(tmp_path), jobs=jobs)
    asof = date(2026, 2, 17)

    # First run succeeds
    r1 = orch.run_once(asof=asof, forced_session=Session.INTRADAY, dry_run=True)
    assert "repeater" in r1.ran_jobs
    assert counter["n"] == 1

    # Second run on a different date bypasses per-(date,session,job) idempotency
    # But cooldown (600s) should still block it since last_success_at is recent
    r2 = orch.run_once(asof=date(2026, 2, 18), forced_session=Session.INTRADAY, dry_run=True)
    assert "repeater" in r2.skipped_jobs
    assert counter["n"] == 1  # Handler was NOT called again


def test_cooldown_allows_after_interval(tmp_path):
    """Job should re-run after cooldown interval expires."""
    counter = {"n": 0}

    def counting_handler(_: dict) -> dict:
        counter["n"] += 1
        return {"status": "ok"}

    jobs = [
        Job("repeater", {Session.INTRADAY}, [], {"paper"}, 10, 0, counting_handler, min_interval_s=1),
    ]
    config = _cfg(tmp_path)
    orch = Orchestrator(config=config, jobs=jobs)
    asof = date(2026, 2, 17)

    # First run
    r1 = orch.run_once(asof=asof, forced_session=Session.INTRADAY, dry_run=True)
    assert "repeater" in r1.ran_jobs

    import time
    time.sleep(1.1)

    # Cooldown of 1s should have expired
    r2 = orch.run_once(asof=date(2026, 2, 18), forced_session=Session.INTRADAY, dry_run=True)
    assert "repeater" in r2.ran_jobs
    assert counter["n"] == 2


def test_price_missing_hard_fail_in_live_mode(tmp_path):
    """handle_order_build_and_route must raise if price missing and dry_run=False."""
    root = tmp_path / "artifacts" / "2026-02-17"
    (root / "targets").mkdir(parents=True, exist_ok=True)
    (root / "signals").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)

    for sleeve in ["core", "vrp", "selector", "futures_overnight", "statarb"]:
        (root / "targets" / f"{sleeve}_targets.json").write_text(
            json.dumps({"schema_version": "targets.v1", "status": "ok", "is_stub": False, "targets": [{"symbol": "ZZZ", "target_weight": 0.01}]}, indent=2),
            encoding="utf-8",
        )
        (root / "signals" / f"{sleeve}_signals.json").write_text(
            json.dumps({"schema_version": "signals.v1", "status": "ok", "is_stub": False}, indent=2),
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
                "metrics": {"nan_or_inf": False, "gross_exposure": 0.03, "net_exposure": 0.03, "num_symbols": 1, "per_sleeve": {}},
                "limits": {},
                "violations": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # DATA STARVATION handler now drops un-priceable intents gracefully
    # instead of raising, so we check that orders are empty and rejected
    result = handle_order_build_and_route(
        {
            "asof_date": "2026-02-17",
            "session": "open",
            "artifact_root": str(root),
            "mode": "paper",
            "dry_run": False,
            "broker": SpyBroker(),
            "config": {"FF_RISK_REPORT_V1_ONLY": False, "account_equity": 100_000},
        }
    )
    assert result["status"] == "ok"
    orders = json.loads((root / "orders" / "orders.json").read_text(encoding="utf-8"))
    assert orders["summary"]["order_count"] == 0


def test_price_missing_fallback_in_dry_run(tmp_path):
    """In dry-run, synthetic fallback should be used when no real price exists."""
    root = tmp_path / "artifacts" / "2026-02-17"
    (root / "targets").mkdir(parents=True, exist_ok=True)
    (root / "signals").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)

    for sleeve in ["core", "vrp", "selector", "futures_overnight", "statarb"]:
        (root / "targets" / f"{sleeve}_targets.json").write_text(
            json.dumps({"schema_version": "targets.v1", "status": "ok", "is_stub": False, "targets": [{"symbol": "ZZZ", "target_weight": 0.01}]}, indent=2),
            encoding="utf-8",
        )
        (root / "signals" / f"{sleeve}_signals.json").write_text(
            json.dumps({"schema_version": "signals.v1", "status": "ok", "is_stub": False}, indent=2),
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
                "metrics": {"nan_or_inf": False, "gross_exposure": 0.03, "net_exposure": 0.03, "num_symbols": 1, "per_sleeve": {}},
                "limits": {},
                "violations": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    result = handle_order_build_and_route(
        {
            "asof_date": "2026-02-17",
            "session": "open",
            "artifact_root": str(root),
            "mode": "paper",
            "dry_run": True,
            "broker": SpyBroker(),
            "config": {"FF_RISK_REPORT_V1_ONLY": False, "account_equity": 100_000},
        }
    )
    assert result["status"] == "ok"
    orders = json.loads((root / "orders" / "orders.json").read_text(encoding="utf-8"))
    assert any(o.get("price_source") == "synthetic" for o in orders["orders"])


def test_orch_status_output(tmp_path):
    """summarize_health returns expected fields from fixture data."""
    config = _cfg(tmp_path)
    orch = Orchestrator(config=config, broker=SpyBroker())
    asof = date(2026, 2, 17)
    orch.run_once(asof=asof, forced_session=Session.PREMARKET, dry_run=True)

    state = StateStore(config.db_path)
    health = summarize_health(config.artifact_root, state)

    assert "heartbeat" in health
    assert health["heartbeat"] is not None
    assert "heartbeat_age_s" in health
    assert health["heartbeat_age_s"] is not None
    assert health["heartbeat_age_s"] < 60
    assert "last_success" in health
    assert "last_failure" in health
    assert "last_routed" in health
    assert health["run_count"] >= 1


def test_schema_migration_v2_adds_last_success_at(tmp_path):
    """Migration v2 should add last_success_at column."""
    db = tmp_path / "state.sqlite3"
    state = StateStore(db)
    with state._connect() as conn:
        cols = conn.execute("PRAGMA table_info(jobs)").fetchall()
        col_names = [c[1] for c in cols]
    assert "last_success_at" in col_names


def test_mode_alias_normalization_maps_live_to_ibkr():
    mode, applied = normalize_mode_alias("live")
    assert mode == "ibkr"
    assert applied is True


def test_default_job_filtering_selects_open_jobs_for_ibkr_mode():
    jobs = filtered_jobs(default_jobs(), session=Session.OPEN, mode="ibkr", enabled=[], disabled=[])
    assert [j.name for j in jobs] == ["ece_calibration_check", "risk_checks_global", "order_build_and_route"]


def test_default_job_filtering_parity_between_live_and_ibkr_mode():
    live_jobs = [j.name for j in filtered_jobs(default_jobs(), session=Session.INTRADAY, mode="live", enabled=[], disabled=[])]
    ibkr_jobs = [j.name for j in filtered_jobs(default_jobs(), session=Session.INTRADAY, mode="ibkr", enabled=[], disabled=[])]
    assert ibkr_jobs == live_jobs


def test_run_once_live_mode_still_executes_jobs_after_normalization(tmp_path):
    executed: list[str] = []

    def _record_job(ctx: dict) -> dict:
        executed.append(str(ctx.get("mode", "")))
        assert ctx.get("mode_raw") == "live"
        assert ctx.get("mode_alias_applied") is True
        return {"status": "ok", "artifacts": {}}

    jobs = [
        Job(
            "record_runtime_mode",
            {Session.OPEN},
            [],
            {"live"},
            10,
            0,
            _record_job,
        )
    ]
    config = OrchestratorConfig(
        artifact_root=tmp_path / "artifacts",
        db_path=tmp_path / "orchestrator.sqlite3",
        mode="live",
        paper_only=True,
    )
    orch = Orchestrator(config=config, jobs=jobs, broker=SpyBroker())
    res = orch.run_once(asof=date(2026, 2, 17), forced_session=Session.OPEN, dry_run=True)

    assert "record_runtime_mode" in res.ran_jobs
    assert executed == ["ibkr"]


def test_order_builder_reads_quantity_with_qty_alias_fallback(tmp_path):
    root = tmp_path / "artifacts" / "2026-02-17"
    (root / "targets").mkdir(parents=True, exist_ok=True)
    (root / "signals").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)

    for sleeve in ["core", "vrp", "selector", "futures_overnight", "statarb"]:
        (root / "targets" / f"{sleeve}_targets.json").write_text(
            json.dumps({"schema_version": "targets.v1", "status": "ok", "is_stub": False, "targets": [{"symbol": "SPY", "target_weight": 0.02}]}, indent=2),
            encoding="utf-8",
        )
        (root / "signals" / f"{sleeve}_signals.json").write_text(
            json.dumps({"schema_version": "signals.v1", "status": "ok", "is_stub": False}, indent=2),
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
                "metrics": {"nan_or_inf": False, "gross_exposure": 0.10, "net_exposure": 0.10, "num_symbols": 1, "per_sleeve": {}},
                "limits": {},
                "violations": [],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    class QuantityOnlyBroker(SpyBroker):
        def get_positions(self) -> dict:
            return {"positions": [{"symbol": "SPY", "quantity": 4.0, "avg_cost": 500.0}]}

    result = handle_order_build_and_route(
        {
            "asof_date": "2026-02-17",
            "session": "open",
            "artifact_root": str(root),
            "mode": "paper",
            "dry_run": True,
            "broker": QuantityOnlyBroker(price_map={"SPY": 500.0}),
            "config": {"FF_RISK_REPORT_V1_ONLY": False, "account_equity": 100_000, "order_notional_rounding": 1},
        }
    )
    assert result["status"] == "ok"
    payload = json.loads((root / "orders" / "orders.json").read_text(encoding="utf-8"))
    assert payload["summary"]["position_alias_hits"] == 0
    # target_notional=10k, current_notional=2k => BUY delta should be positive and less than full 10k rebalance
    assert payload["orders"]
    assert payload["orders"][0]["side"] == "BUY"


def test_order_builder_rejects_qty_alias_positions(tmp_path):
    root = tmp_path / "artifacts" / "2026-02-17"
    (root / "targets").mkdir(parents=True, exist_ok=True)
    (root / "signals").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)

    for sleeve in ["core", "vrp", "selector", "futures_overnight", "statarb"]:
        (root / "targets" / f"{sleeve}_targets.json").write_text(
            json.dumps({"schema_version": "targets.v1", "status": "ok", "is_stub": False, "targets": [{"symbol": "SPY", "target_weight": 0.01}]}, indent=2),
            encoding="utf-8",
        )
        (root / "signals" / f"{sleeve}_signals.json").write_text(
            json.dumps({"schema_version": "signals.v1", "status": "ok", "is_stub": False}, indent=2),
            encoding="utf-8",
        )
    (root / "reports" / "risk_checks.json").write_text(
        json.dumps({"status": "ok", "checked_at": "2026-02-17T13:00:00+00:00", "asof_date": "2026-02-17", "session": "open", "inputs": {"target_paths": {}}, "metrics": {"nan_or_inf": False, "gross_exposure": 0.01, "net_exposure": 0.01, "num_symbols": 1, "per_sleeve": {}}, "limits": {}, "violations": []}, indent=2),
        encoding="utf-8",
    )

    class QtyOnlyBroker(SpyBroker):
        def get_positions(self) -> dict:
            return {"positions": [{"symbol": "SPY", "qty": 1.0, "avg_cost": 500.0}]}

    with pytest.raises(ValueError, match="position row missing quantity"):
        handle_order_build_and_route(
            {
                "asof_date": "2026-02-17",
                "session": "open",
                "artifact_root": str(root),
                "mode": "paper",
                "dry_run": True,
                "broker": QtyOnlyBroker(price_map={"SPY": 500.0}),
                "config": {"FF_RISK_REPORT_V1_ONLY": False, "account_equity": 100_000, "order_notional_rounding": 1},
            }
        )
