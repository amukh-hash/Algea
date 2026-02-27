from backend.app.strategies.statarb.risk import statarb_risk_payload


def test_risk_report_includes_statarb_rl_fields():
    out = statarb_risk_payload("v", "prod", 0.1, 0.1, 1.0, rl_fields={"veto": False})
    assert "rl_policy" in out
