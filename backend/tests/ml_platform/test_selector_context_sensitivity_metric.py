from pathlib import Path

from backend.app.ml_platform.models.selector_smoe.loader import SMoELoader
from backend.app.ml_platform.models.selector_smoe.service import SMoEService
from backend.app.ml_platform.models.selector_smoe.types import SMoERankRequest
from backend.app.ml_platform.registry.store import ModelRegistryStore


def _mk_service(tmp_path: Path) -> SMoEService:
    store = ModelRegistryStore(tmp_path / "registry.db", tmp_path / "models")
    model_dir = tmp_path / "models" / "selector_smoe" / "v1"
    model_dir.mkdir(parents=True)
    (model_dir / "model_config.json").write_text('{"n_experts":4,"top_k":1}', encoding="utf-8")
    (model_dir / "weights.safetensors").write_text("stub", encoding="utf-8")
    store.publish_version("selector_smoe", "v1", "x", {"rank_ic": 0.9, "top_bottom_spread": 0.1, "sharpe": 1.2, "max_drawdown": 0.1, "calibration_score": 0.7}, {"n_experts":4,"top_k":1}, {"feature_schema": {}, "drift_baseline": {}, "calibration": {"calibration_score": 0.7}})
    store.set_alias("selector_smoe", "prod", "v1")
    return SMoEService(SMoELoader(store), tmp_path / "traces")


def test_selector_context_sensitivity_metric(tmp_path: Path):
    svc = _mk_service(tmp_path)
    req = SMoERankRequest(
        asof="2026-01-01",
        symbols=["A", "B"],
        feature_matrix=[[1, 2, 3, 4], [2, 3, 4, 5]],
        market_context={"mkt_ret_5d": 0.01, "mkt_vol_20d": 0.1, "breadth_proxy": 0.5, "sector_rel_strength": 0.02, "volume_weighted_rel_return": 0.03, "vol_regime": 0.12},
        trace_id="t1",
    )
    resp = svc.rank(req)
    assert resp.context_sensitivity_score > 0
