from __future__ import annotations

import json
from datetime import date

from backend.app.orchestrator.calendar import Session
from backend.app.orchestrator.config import OrchestratorConfig
from backend.app.orchestrator.job_defs import Job
from backend.app.orchestrator.orchestrator import Orchestrator


def _emit_models(ctx: dict) -> dict:
    tc = ctx.get("tick_context")
    tc.add_model_version(
        "chronos2",
        model_name="chronos2",
        model_version="chronos-v-test",
        endpoint_name="chronos2_forecast",
        model_alias="prod",
        latency_ms=12.3,
    )
    tc.add_model_version(
        "selector_smoe",
        model_name="selector_smoe",
        model_version="smoe-v-test",
        endpoint_name="smoe_rank",
        model_alias="prod",
        latency_ms=8.1,
    )
    return {"status": "ok"}


def test_orchestrator_records_model_versions_end_to_end(tmp_path):
    cfg = OrchestratorConfig(
        artifact_root=tmp_path / "artifacts",
        db_path=tmp_path / "orch.sqlite3",
        mode="paper",
        paper_only=True,
    )
    jobs = [
        Job("emit_models", {Session.PREMARKET}, [], {"paper", "noop", "live"}, 30, 0, _emit_models),
    ]
    orch = Orchestrator(config=cfg, jobs=jobs)
    res = orch.run_once(asof=date(2026, 2, 17), forced_session=Session.PREMARKET, dry_run=True)
    path = tmp_path / "artifacts" / "2026-02-17" / "model_versions.json"
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["run_id"] == res.run_id
    assert payload["asof_date"] == "2026-02-17"
    assert payload["session"] == "premarket"
    assert payload["models"]["chronos2"]["model_version"] == "chronos-v-test"
    assert payload["models"]["selector_smoe"]["model_version"] == "smoe-v-test"
    for key in ["vol_surface", "vol_surface_grid", "itransformer", "rl_policy"]:
        assert key in payload["models"]
        assert payload["models"][key]["model_version"]
