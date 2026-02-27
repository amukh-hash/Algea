from backend.app.ml_platform.rl.action_projection import project_action


def test_action_projection_veto_and_clamp():
    out, reason = project_action({"size_multiplier": 1.5, "veto": False}, {"max_multiplier": 0.6}, {}, {})
    assert out["size_multiplier"] == 0.6 and reason == "clamped"
    out2, reason2 = project_action({"size_multiplier": 0.8, "veto": True}, {"max_multiplier": 1.0}, {}, {})
    assert out2["size_multiplier"] == 0.0 and reason2 == "veto"
