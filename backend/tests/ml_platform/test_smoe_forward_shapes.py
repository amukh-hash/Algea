from backend.app.ml_platform.models.selector_smoe.model import SMoEConfig, SMoERankerModel


def test_smoe_forward_shape():
    m = SMoERankerModel(SMoEConfig(n_experts=4, top_k=1))
    out = m.forward_row([0.1, 0.2, 0.3], [0.01] * 6)
    assert {"score", "router_entropy", "expert_id", "expert_probs", "uncertainty"}.issubset(out.keys())
    assert len(out["expert_probs"]) == 4
