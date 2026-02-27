from backend.app.ml_platform.rl.env_vrp import VRPSizingEnv


def test_rl_env_determinism():
    e1 = VRPSizingEnv(seed=11)
    e2 = VRPSizingEnv(seed=11)
    s1 = e1.reset()
    s2 = e2.reset()
    assert s1 == s2
    r1 = e1.step({"size_multiplier": 0.5, "veto": False})
    r2 = e2.step({"size_multiplier": 0.5, "veto": False})
    assert r1.reward == r2.reward
    assert r1.state == r2.state
