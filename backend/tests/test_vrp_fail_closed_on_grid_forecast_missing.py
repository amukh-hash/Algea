from backend.app.strategies.vrp.vrp_sleeve import VRPSleeve


class _Client:
    def vol_surface_forecast(self, req, critical=True):
        return type("X", (), {"model_version": "v", "predicted_rv": {7: {"0.50": 0.1}}, "uncertainty": {7: 0.01}, "drift_score": 0.0, "ood_score": 0.0, "latency_ms": 1.0})()


def test_vrp_fail_closed_on_grid_forecast_missing():
    out = VRPSleeve(_Client()).generate_targets("2026-01-01", "SPY", {7: 0.2}, {7: {}}, "t", mode="surface_dynamics", grid_history=None)
    assert out["status"] == "halted"
    assert out["reason"] == "grid_forecast_missing"
