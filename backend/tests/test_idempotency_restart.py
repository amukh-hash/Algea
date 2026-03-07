from __future__ import annotations

import json

from backend.app.orchestrator.job_defs import handle_order_build_and_route
from backend.app.orchestrator.broker import PaperBrokerStub


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_deterministic_client_order_ids_same_tick(tmp_path):
    root = tmp_path / "artifacts" / "2026-02-17"
    _write(root / "reports" / "risk_checks.json", {"status": "ok", "violations": [], "metrics": {}, "limits": {}})
    for sleeve in ["core", "vrp", "selector", "futures_overnight", "statarb"]:
        _write(root / "signals" / f"{sleeve}_signals.json", {"schema_version": "signals.v1", "status": "ok", "is_stub": False})
        _write(root / "targets" / f"{sleeve}_targets.json", {"schema_version": "targets.v1", "status": "ok", "is_stub": False, "targets": [{"symbol": "SPY", "target_weight": 0.01}]})

    ctx = {
        "asof_date": "2026-02-17",
        "session": "open",
        "artifact_root": str(root),
        "mode": "paper",
        "dry_run": True,
        "tick_id": "tick-1",
        "broker": PaperBrokerStub(price_map={"SPY": 500.0}),
        "config": {"account_equity": 100_000},
        "control_snapshot": {"snapshot_id": "snap-1", "paused": False},
    }
    handle_order_build_and_route(ctx)
    first = json.loads((root / "orders" / "orders.json").read_text(encoding="utf-8"))
    handle_order_build_and_route(ctx)
    second = json.loads((root / "orders" / "orders.json").read_text(encoding="utf-8"))
    ids1 = [o["client_order_id"] for o in first["orders"]]
    ids2 = [o["client_order_id"] for o in second["orders"]]
    assert ids1 == ids2
