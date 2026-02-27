from backend.app.strategies.vrp.vrp_decision import choose_tenor


def test_vrp_edge_gating_determinism():
    edge = {7: 0.02, 30: 0.03}
    unc = {7: 0.1, 30: 0.1}
    assert choose_tenor(edge, unc, 0.01, 0.2) == 30
