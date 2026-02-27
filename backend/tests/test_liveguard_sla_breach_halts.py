from datetime import date

from backend.app.risk.live_guard import LiveGuard


def test_liveguard_sla_breach_halts():
    d = LiveGuard().evaluate(
        date(2026, 1, 1),
        scenario_loss_pct=0.0,
        margin_utilization=0.1,
        forecast_health=0.9,
        sla_breach=True,
    )
    assert d.status == "halted"
    assert not d.allow_new_trades
    assert "halt_sla_breach" in d.reasons
