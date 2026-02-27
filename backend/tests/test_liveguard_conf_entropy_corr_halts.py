from datetime import date

from backend.app.risk.live_guard import LiveGuard


def test_liveguard_conf_entropy_corr_halts():
    d = LiveGuard().evaluate(date(2026, 1, 1), scenario_loss_pct=0.0, margin_utilization=0.1, forecast_health=0.8, confidence_series=[0.1, 0.2, 0.3], outcome_series=[3.0, 2.0, 1.0], corr_baseline=0.9)
    assert not d.allow_new_trades
    assert "halt_conf_entropy_corr" in d.reasons
