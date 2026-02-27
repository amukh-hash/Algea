from backend.app.ml_platform.feature_store.labels_vol import forward_realized_vol


def test_vol_labels_no_lookahead():
    close = [100, 101, 102]
    assert forward_realized_vol(close, 2, 1) is None
