from datetime import date

from backend.app.risk.live_guard import LiveGuard


def test_liveguard_drift_halts():
    d = LiveGuard().evaluate(date(2026, 1, 1), scenario_loss_pct=0.0, margin_utilization=0.1, forecast_health=0.8, prediction_series=[10, 10, 10], consistency_baseline={"pred_mean": 0.0, "pred_std": 1.0})
    assert not d.allow_new_trades
    assert "halt_prediction_consistency" in d.reasons
