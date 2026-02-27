import json

from backend.app.orchestrator.job_defs import handle_risk_checks_global


def _w(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_risk_report_includes_smoe_fields(tmp_path):
    for sleeve in ["core", "vrp", "selector"]:
        ml = {"router_entropy_mean": 0.2, "expert_utilization": {"0": 10}, "load_balance_score": 0.1}
        _w(tmp_path / "targets" / f"{sleeve}_targets.json", {"schema_version": "targets.v1", "status": "ok", "targets": [], "ml_risk": ml})
    handle_risk_checks_global({"asof_date": "2026-01-01", "session": "OPEN", "artifact_root": str(tmp_path), "config": {}})
    report = json.loads((tmp_path / "reports" / "risk_checks.json").read_text(encoding="utf-8"))
    assert "router_entropy_mean" in report["metrics"]["per_sleeve"]["selector"]["ml"]
