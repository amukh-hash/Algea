from backend.app.ml_platform.feature_store.builders.multivariate_panel import build_multivariate_panel_dataset


def test_panel_builder_determinism(tmp_path):
    close = {"XLF": [100 + i for i in range(100)], "XLK": [200 + i for i in range(100)]}
    d1 = build_multivariate_panel_dataset("p1", "2026-01-01", ["XLK", "XLF"], close, idx=50, out_root=tmp_path)
    d2 = build_multivariate_panel_dataset("p1", "2026-01-01", ["XLF", "XLK"], close, idx=50, out_root=tmp_path)
    assert d1["manifest"]["manifest_hash"] == d2["manifest"]["manifest_hash"]
    assert [r["symbol"] for r in d1["rows"]] == ["XLF", "XLK"]
