from backend.app.ml_platform.models.selector_smoe.model import SMoERankerModel


def test_selector_router_uses_context():
    m = SMoERankerModel()
    a = m.forward_row([1, 1, 1, 1], [0.0, 0.0])
    b = m.forward_row([1, 1, 1, 1], [1.0, 1.0])
    assert a["score"] != b["score"]
