from backend.app.ml_platform.config import MLPlatformConfig


def test_rl_canary_alias_wiring():
    cfg = MLPlatformConfig(rl_policy_alias_vrp="canary", rl_policy_alias_statarb="staging")
    assert cfg.rl_policy_alias_vrp == "canary"
    assert cfg.rl_policy_alias_statarb == "staging"
