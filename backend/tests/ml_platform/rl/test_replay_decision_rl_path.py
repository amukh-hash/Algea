import json

from backend.app.ml_platform.models.rl_policy.types import RLPolicyResponse
from backend.app.ml_platform.replay.replay_decision import replay_rl_policy_decision


class _Client:
    def rl_policy_act(self, req, critical=True):
        return RLPolicyResponse(model_version="v1", size_multiplier=0.5, veto=False, projected_multiplier=0.5, projection_reason="clamped", projection_applied=True, drift_score=0.0, ood_score=0.0, latency_ms=1.0)


def test_replay_decision_rl_path(tmp_path):
    p = tmp_path / "trace.json"
    p.write_text(json.dumps({"request_payload": {"asof": "2026-01-01", "sleeve": "vrp", "state": {"x": 0.1}, "proposal": {"base_size": 0.02}, "constraints": {"max_multiplier": 1.0}, "model_alias": "prod", "trace_id": "t"}}), encoding="utf-8")
    out = replay_rl_policy_decision(p, _Client())
    assert out["model_version"] == "v1"
