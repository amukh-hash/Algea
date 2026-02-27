from backend.app.ml_platform.models.selector_smoe.router import topk_router


def test_topk_router_invariants():
    chosen, probs, ent = topk_router([1.0, 2.0, 0.5], k=2)
    assert len(chosen) == 2
    assert abs(sum(probs) - 1.0) < 1e-6
    assert ent > 0
