import json
from pathlib import Path

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
from backend.app.ml_platform.inference_gateway.protocol import InferenceRequestBase
from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_trace_written_for_chronos_call(tmp_path: Path) -> None:
    cfg = MLPlatformConfig(registry_db_path=tmp_path / "r.sqlite", model_root=tmp_path / "models", trace_root=tmp_path / "traces")
    store = ModelRegistryStore(cfg.registry_db_path, cfg.model_root)
    store.publish_version("chronos2", "v1", "a", {"pinball_loss": 0.1}, {"f": 1}, {"feature_schema": {}, "drift_baseline": {"mean": 0, "std": 1}, "calibration": {"calibration_score": 0.8}})
    store.set_alias("chronos2", "prod", "v1")
    server = InferenceGatewayServer(cfg)
    wrapped = InferenceRequestBase(
        asof=__import__("datetime").datetime(2026, 1, 1),
        universe_id="ES",
        features_hash="h",
        model_alias="prod",
        trace_id="trace-123",
        payload={"asof": "2026-01-01", "series": [1, 2, 3, 4], "freq": "1d", "prediction_length": 2, "quantiles": [0.1, 0.5, 0.9], "instrument_id": "ES", "trace_id": "trace-123", "model_alias": "prod"},
    )
    server.infer("chronos2_forecast", wrapped)
    trace_path = cfg.trace_root / "trace-123.json"
    assert trace_path.exists()
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["model_version"] == "v1"
    assert "request_hash" in payload and "output_hash" in payload
