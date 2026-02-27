from backend.app.strategies.vrp.vrp_sleeve import VRPSleeve


class _Client:
    def vol_surface_forecast(self, req, critical=True):
        return type("X", (), {"model_version": "v", "predicted_rv": {7: {"0.50": 0.1}}, "uncertainty": {7: 0.01}, "drift_score": 0.0, "ood_score": 0.0, "latency_ms": 1.0})()

    def vol_surface_grid_forecast(self, req, critical=True):
        return type("G", (), {"grid_forecast": {"7:ATM": 0.3}, "uncertainty_proxy": 0.1, "mask_coverage": 0.9})()

    def rl_policy_act(self, req, critical=True):
        return type("R", (), {"size_multiplier": 1.0, "veto": False, "model_version": "r1", "latency_ms": 1.0, "drift_score": 0.0, "ood_score": 0.0})()


def test_risk_report_includes_vrp_grid_fields():
    out = VRPSleeve(_Client()).generate_targets("2026-01-01", "SPY", {7: 0.2}, {7: {}}, "t", mode="surface_dynamics", grid_history=[{"iv": {"7:ATM": 0.2}}])
    assert "grid_deltas" in out["ml_risk"]
