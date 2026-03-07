import pytest
from pathlib import Path

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
from backend.app.ml_platform.models.rl_policy.artifact import save_rl_policy_artifact
from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_health_includes_rl_policy(tmp_path: Path):
    cfg = MLPlatformConfig(registry_db_path=tmp_path / "r.sqlite", model_root=tmp_path / "models", trace_root=tmp_path / "traces")
    store = ModelRegistryStore(cfg.registry_db_path, cfg.model_root)
    out = tmp_path / "artifact"
    out.mkdir(parents=True, exist_ok=True)
    import torch; torch.save({"_test": True}, out / "weights.safetensors")
    save_rl_policy_artifact(out, {"hidden_size": 8}, {"mean_return": 0.1}, {"state_mean": 0.0})
    store.publish_artifact_directory("rl_policy", "v1", out, "x", {"mean_return": 0.1}, {"hidden_size": 8}, {"feature_schema": {}, "drift_baseline": {"state_mean": 0.0}, "calibration": {"ece": 0.1}})
    store.set_alias("rl_policy", "prod", "v1")
    s = InferenceGatewayServer(cfg)
    assert "rl_policy:prod" in s.get_health()["models"]
