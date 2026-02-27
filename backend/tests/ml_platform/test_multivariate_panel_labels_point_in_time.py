from backend.app.ml_platform.feature_store.panel_labels import panel_label_fwd_ret


def test_panel_labels_pit():
    close = [100, 101, 102]
    assert panel_label_fwd_ret(close, 2, horizon=1) is None
