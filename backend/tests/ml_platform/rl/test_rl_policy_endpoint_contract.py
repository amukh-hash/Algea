from pathlib import Path

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.inference_gateway.server import InferenceGatewayServer
from backend.app.ml_platform.models.rl_policy.artifact import save_rl_policy_artifact
from backend.app.ml_platform.models.rl_policy.types import RLPolicyRequest
from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_rl_policy_endpoint_contract(tmp_path: Path):
    cfg = MLPlatformConfig(registry_db_path=tmp_path / "r.sqlite", model_root=tmp_path / "models", trace_root=tmp_path / "traces")
    store = ModelRegistryStore(cfg.registry_db_path, cfg.model_root)
    out = tmp_path / "artifact"
    save_rl_policy_artifact(out, {"hidden_size": 8}, {"mean_return": 0.1}, {"state_mean": 0.0})
    store.publish_artifact_directory("rl_policy", "v1", out, "x", {"mean_return": 0.1}, {"hidden_size": 8}, {"feature_schema": {}, "drift_baseline": {"state_mean": 0.0}, "calibration": {"ece": 0.1}})
    store.set_alias("rl_policy", "prod", "v1")
    s = InferenceGatewayServer(cfg)
    payload = s.rl_policy_http_act(RLPolicyRequest(asof="2026-01-01", sleeve="vrp", state={"a": 0.1}, proposal={"base_size": 0.02}, constraints={"max_multiplier": 1.0}, trace_id="rl1"))
    assert payload["model_name"] == "rl_policy"
    assert "size_multiplier" in payload
