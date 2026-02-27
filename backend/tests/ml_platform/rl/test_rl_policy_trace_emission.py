import json
from pathlib import Path

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.models.rl_policy.artifact import save_rl_policy_artifact
from backend.app.ml_platform.models.rl_policy.loader import RLPolicyLoader
from backend.app.ml_platform.models.rl_policy.service import RLPolicyService
from backend.app.ml_platform.models.rl_policy.types import RLPolicyRequest
from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_rl_policy_trace_emission(tmp_path: Path):
    cfg = MLPlatformConfig(registry_db_path=tmp_path / "r.sqlite", model_root=tmp_path / "models", trace_root=tmp_path / "traces")
    store = ModelRegistryStore(cfg.registry_db_path, cfg.model_root)
    out = tmp_path / "artifact"
    save_rl_policy_artifact(out, {"hidden_size": 8}, {"mean_return": 0.1}, {"state_mean": 0.0})
    store.publish_artifact_directory("rl_policy", "v1", out, "x", {"mean_return": 0.1}, {"hidden_size": 8}, {"feature_schema": {}, "drift_baseline": {"state_mean": 0.0}, "calibration": {"ece": 0.1}})
    store.set_alias("rl_policy", "prod", "v1")
    svc = RLPolicyService(RLPolicyLoader(store), cfg.trace_root)
    svc.policy_act(RLPolicyRequest(asof="2026-01-01", sleeve="vrp", state={"x": 0.2}, proposal={"base_size": 0.02}, constraints={"max_multiplier": 1.0}, trace_id="rl-tr"))
    payload = json.loads((cfg.trace_root / "rl-tr.json").read_text(encoding="utf-8"))
    assert "projection_reason" in payload and "state_hash" in payload
