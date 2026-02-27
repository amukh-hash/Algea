from backend.app.strategies.vrp.vrp_risk import build_vrp_ml_risk


def test_risk_report_includes_vrp_rl_fields():
    out = build_vrp_ml_risk("v", "prod", {}, {}, {}, 0.0, 0.0, 1.0, rl_fields={"veto": False})
    assert "rl_policy" in out and "veto" in out["rl_policy"]
