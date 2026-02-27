from pathlib import Path

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_alias_load_drives_readyz(tmp_path: Path) -> None:
    cfg = MLPlatformConfig(
        registry_db_path=tmp_path / "registry.sqlite",
        model_root=tmp_path / "models",
        trace_root=tmp_path / "traces",
    )
    store = ModelRegistryStore(cfg.registry_db_path, cfg.model_root)
    store.publish_version(
        "chronos2",
        "v9",
        "abc",
        metrics={"pinball_loss": 0.1, "calibration_score": 0.8},
        config={"freq": "1d"},
        data_lineage={"feature_schema": {"name": "TSFM"}, "drift_baseline": {"mean": 100.0, "std": 2.0}, "calibration": {"calibration_score": 0.8}},
    )
    store.set_alias("chronos2", "prod", "v9")

    server = InferenceGatewayServer(cfg)
    assert server.get_ready()["ready"] is True
    models = server.list_models()
    assert models["chronos2:prod"]["status"] == "loaded"
