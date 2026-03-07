import json

from backend.app.orchestrator.job_defs import handle_risk_checks_global


def _w(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_risk_report_includes_statarb_fields(tmp_path):
    _w(tmp_path / "targets" / "core_targets.json", {"schema_version": "targets.v1", "status": "ok", "targets": []})
    _w(tmp_path / "targets" / "vrp_targets.json", {"schema_version": "targets.v1", "status": "ok", "targets": []})
    _w(tmp_path / "targets" / "selector_targets.json", {"schema_version": "targets.v1", "status": "ok", "targets": []})
    _w(tmp_path / "targets" / "futures_overnight_targets.json", {"schema_version": "targets.v1", "status": "ok", "targets": []})
    _w(tmp_path / "targets" / "statarb_targets.json", {"schema_version": "targets.v1", "status": "ok", "targets": [], "ml_risk": {"corr_break_score": 0.1, "beta_residual": 0.0}})
    handle_risk_checks_global({"asof_date": "2026-01-01", "session": "OPEN", "artifact_root": str(tmp_path), "config": {"enable_statarb_sleeve": True}})
    report = json.loads((tmp_path / "reports" / "risk_checks.json").read_text(encoding="utf-8"))
    assert "corr_break_score" in report["metrics"]["per_sleeve"]["statarb"]["ml"]
