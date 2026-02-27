from __future__ import annotations

import json

from backend.app.orchestrator.job_defs import handle_risk_checks_global


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_risk_report_ml_fields_present(tmp_path):
    root = tmp_path
    for sleeve in ["core", "vrp", "selector"]:
        _write(root / "targets" / f"{sleeve}_targets.json", {"schema_version": "targets.v1", "status": "ok", "targets": [], "ml_risk": {"model_version": "na"}})
    out = handle_risk_checks_global({"asof_date": "2026-01-01", "session": "OPEN", "artifact_root": str(root), "config": {}})
    report = json.loads((root / "reports" / "risk_checks.json").read_text(encoding="utf-8"))
    assert out["status"] == "ok"
    assert "ml" in report["metrics"]["per_sleeve"]["core"]
