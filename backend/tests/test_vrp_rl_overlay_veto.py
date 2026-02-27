from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.models.rl_policy.types import RLPolicyResponse
from backend.app.ml_platform.models.vol_surface.types import VolSurfaceResponse
from backend.app.strategies.vrp.vrp_sleeve import VRPSleeve


class _Client:
    def vol_surface_forecast(self, req, critical=True):
        return VolSurfaceResponse(model_version="v", predicted_rv={7: {"0.50": 0.1}}, uncertainty={7: 0.1}, ood_score=0.0, drift_score=0.0, latency_ms=1.0)

    def rl_policy_act(self, req, critical=True):
        return RLPolicyResponse(model_version="r1", size_multiplier=0.5, veto=True, projected_multiplier=0.0, projection_reason="veto", projection_applied=True, drift_score=0.0, ood_score=0.0, latency_ms=1.0)


def test_vrp_rl_overlay_veto():
    cfg = MLPlatformConfig(enable_rl_overlay_vrp=True)
    out = VRPSleeve(_Client(), cfg=cfg).generate_targets("2026-01-01", "SPY", {7: 0.14}, {7: {"rv_hist_20": 0.2}}, "t")
    assert out["targets"] == []
