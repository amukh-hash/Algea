from backend.app.ml_platform.feature_store.labels import fwd_return


def test_labels_no_future_leakage():
    prices = [100, 101, 102, 103]
    assert fwd_return(prices, 2, 2) is None
    assert fwd_return(prices, 0, 1) == (101 / 100) - 1
