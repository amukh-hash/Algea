from datetime import date

from backend.app.risk.live_guard import LiveGuard


def test_liveguard_ece_halts():
    d = LiveGuard().evaluate(date(2026, 1, 1), scenario_loss_pct=0.0, margin_utilization=0.1, forecast_health=0.8, current_confidence=0.9, current_accuracy=0.1)
    assert not d.allow_new_trades
    assert "halt_ece_spike" in d.reasons
