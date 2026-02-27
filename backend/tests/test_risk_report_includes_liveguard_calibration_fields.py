from datetime import date

from backend.app.risk.live_guard import LiveGuard


def test_risk_report_includes_liveguard_calibration_fields():
    d = LiveGuard().evaluate(date(2026, 1, 1), scenario_loss_pct=0.0, margin_utilization=0.1, forecast_health=0.8)
    assert "ece" in d.metrics and "prediction_consistency_score" in d.metrics and "conf_entropy_corr" in d.metrics
    assert d.status in {"ok", "halted"}
