import json
from pathlib import Path

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.models.selector_smoe.loader import SMoELoader
from backend.app.ml_platform.models.selector_smoe.service import SMoEService
from backend.app.ml_platform.models.selector_smoe.types import SMoERankRequest
from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_smoe_trace_fields(tmp_path: Path):
    cfg = MLPlatformConfig(registry_db_path=tmp_path / "r.sqlite", model_root=tmp_path / "models", trace_root=tmp_path / "traces")
    store = ModelRegistryStore(cfg.registry_db_path, cfg.model_root)
    store.publish_version("selector_smoe", "v1", "x", {"rank_ic": 0.6, "calibration_score": 0.7}, {"n_experts": 4, "top_k": 1}, {"feature_schema": {}, "drift_baseline": {}, "calibration": {"calibration_score": 0.7}})
    store.set_alias("selector_smoe", "prod", "v1")
    svc = SMoEService(SMoELoader(store), cfg.trace_root)
    svc.rank(SMoERankRequest(asof="2026-01-01", symbols=["A"], feature_matrix=[[0.1]], trace_id="smoe-tr", market_context={}))
    payload = json.loads((cfg.trace_root / "smoe-tr.json").read_text(encoding="utf-8"))
    assert "router_entropy_mean" in payload and "expert_utilization" in payload
