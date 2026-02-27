from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.models.itransformer.types import ITransformerSignalResponse
from backend.app.ml_platform.models.rl_policy.types import RLPolicyResponse
from backend.app.strategies.statarb.sleeve import StatArbSleeve


class _Client:
    def itransformer_signal(self, req, critical=True):
        return ITransformerSignalResponse(model_version="i1", scores={"A": 1.0, "B": -1.0}, uncertainty=0.2, correlation_regime=0.5, latency_ms=1.0)

    def rl_policy_act(self, req, critical=True):
        return RLPolicyResponse(model_version="r1", size_multiplier=0.4, veto=False, projected_multiplier=0.4, projection_reason="clamped", projection_applied=True, drift_score=0.0, ood_score=0.0, latency_ms=1.0)


def test_statarb_rl_overlay_scaling():
    out = StatArbSleeve(_Client(), cfg=MLPlatformConfig(enable_rl_overlay_statarb=True)).generate_targets("2026-01-01", ["A", "B"], [[1], [2]], "t")
    assert out["status"] == "ok"
    assert max(abs(t["target_weight"]) for t in out["targets"]) < 1.0
