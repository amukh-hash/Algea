from pathlib import Path

from backend.app.ml_platform.registry.promotion import promote_if_eligible
from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_selector_promotion_gates_context_and_collapse(tmp_path: Path):
    store = ModelRegistryStore(tmp_path / "registry.db", tmp_path / "models")
    ok = promote_if_eligible(
        store,
        "selector_smoe",
        "v1",
        {"sharpe": 1.1, "max_drawdown": 0.1, "calibration_score": 0.8, "rank_ic": 0.8, "top_bottom_spread": 0.1, "router_entropy_mean": 0.1, "expert_collapse_score": 0.6, "context_sensitivity_score": 0.05},
    )
    assert ok
    bad = promote_if_eligible(
        store,
        "selector_smoe",
        "v2",
        {"sharpe": 1.1, "max_drawdown": 0.1, "calibration_score": 0.8, "rank_ic": 0.8, "top_bottom_spread": 0.1, "router_entropy_mean": 0.1, "expert_collapse_score": 0.95, "context_sensitivity_score": 0.0},
    )
    assert not bad
