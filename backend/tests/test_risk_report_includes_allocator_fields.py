from __future__ import annotations

import json

from backend.app.orchestrator.job_defs import handle_risk_checks_global


def _w(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_risk_report_includes_allocator_fields(tmp_path):
    _w(tmp_path / "targets" / "core_targets.json", {"schema_version": "targets.v1", "status": "ok", "targets": [{"symbol": "SPY", "target_weight": 0.04}], "ml_risk": {"expected_return_proxy": 0.1, "uncertainty": 0.01}})
    _w(tmp_path / "targets" / "vrp_targets.json", {"schema_version": "targets.v1", "status": "ok", "targets": [{"symbol": "SPY", "target_weight": 0.03}], "ml_risk": {"edge_mean": 0.2, "drift_score": 0.01}})
    _w(tmp_path / "targets" / "selector_targets.json", {"schema_version": "targets.v1", "status": "ok", "targets": [{"symbol": "AAPL", "target_weight": 0.04}], "ml_risk": {"top_bottom_spread": 0.05, "router_entropy_mean": 0.02}})
    _w(tmp_path / "targets" / "futures_overnight_targets.json", {"schema_version": "targets.v1", "status": "ok", "targets": []})
    _w(tmp_path / "targets" / "statarb_targets.json", {"schema_version": "targets.v1", "status": "ok", "targets": []})

    ctx = {
        "asof_date": "2026-01-01",
        "session": "OPEN",
        "artifact_root": str(tmp_path),
        "config": {"enable_allocator": True, "max_gross": 1.5},
    }
    handle_risk_checks_global(ctx)
    report1 = json.loads((tmp_path / "reports" / "risk_checks.json").read_text(encoding="utf-8"))
    handle_risk_checks_global(ctx)
    report2 = json.loads((tmp_path / "reports" / "risk_checks.json").read_text(encoding="utf-8"))

    assert report1["allocator"]["enabled"] is True
    assert report1["allocator"]["status"] == "ok"
    assert set(report1["allocator"]["inputs"].keys()) >= {"core", "vrp", "selector"}
    assert set(report1["allocator"]["outputs"].keys()) >= {"core", "vrp", "selector"}
    assert report1["allocator"]["constraints"]["total_gross_cap"] == 1.5
    assert report1["allocator"]["outputs"] == report2["allocator"]["outputs"]
