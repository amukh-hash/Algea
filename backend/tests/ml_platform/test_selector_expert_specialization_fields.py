from pathlib import Path

from backend.app.ml_platform.models.selector_smoe.loader import SMoELoader
from backend.app.ml_platform.models.selector_smoe.service import SMoEService
from backend.app.ml_platform.models.selector_smoe.types import SMoERankRequest
from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_selector_expert_specialization_fields(tmp_path: Path):
    store = ModelRegistryStore(tmp_path / "registry.db", tmp_path / "models")
    model_dir = tmp_path / "models" / "selector_smoe" / "v1"
    model_dir.mkdir(parents=True)
    (model_dir / "model_config.json").write_text('{"n_experts":4,"top_k":1}', encoding="utf-8")
    (model_dir / "weights.safetensors").write_text("stub", encoding="utf-8")
    store.publish_version("selector_smoe", "v1", "x", {"rank_ic": 0.9, "top_bottom_spread": 0.2, "sharpe": 1.2, "max_drawdown": 0.1, "calibration_score": 0.9}, {"n_experts":4,"top_k":1}, {"feature_schema": {}, "drift_baseline": {}, "calibration": {"calibration_score": 0.9}})
    store.set_alias("selector_smoe", "prod", "v1")
    svc = SMoEService(SMoELoader(store), tmp_path / "traces")
    resp = svc.rank(SMoERankRequest(asof="2026-01-01", symbols=["A"], feature_matrix=[[1,1,1,1]], market_context={"mkt_ret_5d":0.1,"mkt_vol_20d":0.2,"breadth_proxy":0.6,"sector_rel_strength":0.1,"volume_weighted_rel_return":0.1,"vol_regime":0.2}, trace_id="t"))
    assert resp.specialization_by_bucket
    assert resp.expert_collapse_score >= 0
