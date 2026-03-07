from __future__ import annotations

import json
from datetime import date

import pytest

from backend.app.orchestrator.job_defs import (
    _generic_signal_handler,
    handle_order_build_and_route,
)
from backend.app.orchestrator.broker import PaperBrokerStub


def _ctx(tmp_path, mode="paper", dry_run=False):
    return {
        "asof_date": "2026-02-17",
        "session": "OPEN",
        "artifact_root": str(tmp_path),
        "mode": mode,
        "dry_run": dry_run,
        "broker": PaperBrokerStub(),
        "config": {"account_equity": 100000, "max_orders": 20, "max_order_notional": 100000, "max_total_order_notional": 500000},
    }


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_stub_signal_blocked_outside_noop(tmp_path, monkeypatch):
    monkeypatch.delenv("ORCH_ALLOW_STUB_SIGNALS", raising=False)
    with pytest.raises(RuntimeError):
        _generic_signal_handler(_ctx(tmp_path, mode="paper"), "core", ["SPY"])


def test_route_gate_rejects_stub_signal_artifact(tmp_path):
    ctx = _ctx(tmp_path)
    root = tmp_path
    _write(root / "reports" / "risk_checks.json", {"status": "ok", "violations": [], "metrics": {}, "limits": {}})
    for sleeve in ["core", "vrp", "selector", "futures_overnight", "statarb"]:
        _write(root / "signals" / f"{sleeve}_signals.json", {"schema_version": "signals.v1", "status": "ok", "is_stub": sleeve == "core"})
        _write(root / "targets" / f"{sleeve}_targets.json", {"schema_version": "targets.v1", "status": "ok", "is_stub": False, "targets": []})

    with pytest.raises(RuntimeError, match="signal artifact invalid|cannot build orders"):
        handle_order_build_and_route(ctx)


def test_empty_orders_is_success(tmp_path):
    ctx = _ctx(tmp_path)
    root = tmp_path
    _write(root / "reports" / "risk_checks.json", {"status": "ok", "violations": [], "metrics": {}, "limits": {}})
    for sleeve in ["core", "vrp", "selector", "futures_overnight", "statarb"]:
        _write(root / "signals" / f"{sleeve}_signals.json", {"schema_version": "signals.v1", "status": "ok", "is_stub": False})
        _write(root / "targets" / f"{sleeve}_targets.json", {"schema_version": "targets.v1", "status": "ok", "is_stub": False, "targets": []})

    out = handle_order_build_and_route(ctx)
    assert out["status"] == "ok"
    assert out["summary"]["order_count"] == 0
