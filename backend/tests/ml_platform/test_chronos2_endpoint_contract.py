from pathlib import Path

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
from backend.app.ml_platform.models.chronos2.types import TSFMRequest
from backend.app.ml_platform.registry.store import ModelRegistryStore


def _seed_registry(tmp_path: Path) -> MLPlatformConfig:
    cfg = MLPlatformConfig(
        registry_db_path=tmp_path / "registry.sqlite",
        model_root=tmp_path / "models",
        trace_root=tmp_path / "traces",
    )
    store = ModelRegistryStore(cfg.registry_db_path, cfg.model_root)
    store.publish_version(
        "chronos2",
        "v1",
        "abc",
        metrics={"pinball_loss": 0.1, "calibration_score": 0.8},
        config={"freq": "1d"},
        data_lineage={"feature_schema": {"name": "TSFM"}, "drift_baseline": {"mean": 100.0, "std": 2.0}, "calibration": {"calibration_score": 0.8}},
    )
    store.set_alias("chronos2", "prod", "v1")
    return cfg


def test_chronos2_contract_shape(tmp_path: Path) -> None:
    cfg = _seed_registry(tmp_path)
    server = InferenceGatewayServer(cfg)
    data = server.chronos2_http_forecast(
        TSFMRequest(
            asof="2026-01-02",
            series=[100, 101, 102, 103],
            freq="1d",
            prediction_length=3,
            quantiles=[0.1, 0.5, 0.9],
            instrument_id="ES",
            trace_id="t-1",
            model_alias="prod",
        )
    )
    assert data["model_name"] == "chronos2"
    assert set(data["forecast"].keys()) == {"0.10", "0.50", "0.90"}
    assert len(data["forecast"]["0.50"]) == 3
