import json

from backend.app.orchestrator.job_defs import handle_risk_checks_global


def _w(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_risk_report_includes_vrp_surface_fields(tmp_path):
    _w(tmp_path / "targets" / "core_targets.json", {"schema_version": "targets.v1", "status": "ok", "targets": []})
    _w(tmp_path / "targets" / "selector_targets.json", {"schema_version": "targets.v1", "status": "ok", "targets": []})
    _w(tmp_path / "targets" / "vrp_targets.json", {"schema_version": "targets.v1", "status": "ok", "targets": [], "ml_risk": {"edge_by_tenor": {"7": 0.01}, "predicted_rv": {"7": {"0.50": 0.2}}}})
    _w(tmp_path / "targets" / "futures_overnight_targets.json", {"schema_version": "targets.v1", "status": "ok", "targets": []})
    _w(tmp_path / "targets" / "statarb_targets.json", {"schema_version": "targets.v1", "status": "ok", "targets": []})
    handle_risk_checks_global({"asof_date": "2026-01-01", "session": "OPEN", "artifact_root": str(tmp_path), "config": {}})
    report = json.loads((tmp_path / "reports" / "risk_checks.json").read_text(encoding="utf-8"))
    assert "edge_by_tenor" in report["metrics"]["per_sleeve"]["vrp"]["ml"]
