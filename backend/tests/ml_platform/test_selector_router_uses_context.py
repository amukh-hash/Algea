from backend.app.ml_platform.models.selector_smoe.model import SMoERankerModel
from backend.app.ml_platform.models.selector_master.context_encoder import validate_market_context


def test_selector_router_uses_context():
    m = SMoERankerModel()
    a = m.forward_row([1, 1, 1, 1], [0.0, 0.0])
    b = m.forward_row([1, 1, 1, 1], [1.0, 1.0])
    assert a["score"] != b["score"]


def test_selector_context_required_semantics():
    try:
        validate_market_context({"mkt_ret_5d": 0.1})
    except ValueError:
        return
    assert False, "missing context should fail fast"
