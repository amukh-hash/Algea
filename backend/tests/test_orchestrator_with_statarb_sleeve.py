from __future__ import annotations

from datetime import date

from backend.app.ml_platform.registry.store import ModelRegistryStore
from backend.app.orchestrator.calendar import Session
from backend.app.orchestrator.config import OrchestratorConfig
from backend.app.orchestrator.orchestrator import Orchestrator


def test_orchestrator_runs_with_statarb_sleeve(tmp_path, monkeypatch):
    monkeypatch.setenv("ORCH_ALLOW_STUB_SIGNALS", "1")
    monkeypatch.setenv("ENABLE_STATARB_SLEEVE", "1")
    db = tmp_path / "reg.sqlite"
    model_root = tmp_path / "models"
    trace_root = tmp_path / "traces"
    monkeypatch.setenv("MODEL_REGISTRY_DB", str(db))
    monkeypatch.setenv("MODEL_ARTIFACT_ROOT", str(model_root))
    monkeypatch.setenv("TRACE_ROOT", str(trace_root))

    store = ModelRegistryStore(db, model_root)
    store.publish_version("itransformer", "v1", "x", {"rank_ic": 0.7, "pair_stability": 0.8, "calibration_score": 0.8, "sharpe": 1.1, "max_drawdown": 0.1}, {"hidden_size": 16}, {"feature_schema": {}, "drift_baseline": {"score_mean": 0.0}, "calibration": {"calibration_score": 0.8}})
    store.set_alias("itransformer", "prod", "v1")

    cfg = OrchestratorConfig(
        artifact_root=tmp_path / "artifacts",
        db_path=tmp_path / "orch.sqlite",
        mode="noop",
        enable_statarb_sleeve=True,
    )
    orch = Orchestrator(config=cfg)
    res = orch.run_once(asof=date(2026, 2, 17), forced_session=Session.PREMARKET, dry_run=True)
    assert "signals_generate_statarb" in res.ran_jobs
