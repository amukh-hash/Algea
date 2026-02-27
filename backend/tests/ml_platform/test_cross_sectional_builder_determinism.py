from backend.app.ml_platform.feature_store.builders.cross_sectional import build_cross_sectional_dataset


def test_cross_sectional_builder_determinism(tmp_path):
    close = {"AAPL": [100 + i for i in range(40)], "MSFT": [200 + i * 0.5 for i in range(40)]}
    ds1 = build_cross_sectional_dataset("d1", "2026-01-01", ["MSFT", "AAPL"], close, out_root=tmp_path)
    ds2 = build_cross_sectional_dataset("d1", "2026-01-01", ["AAPL", "MSFT"], close, out_root=tmp_path)
    assert ds1["manifest"]["manifest_hash"] == ds2["manifest"]["manifest_hash"]
    assert [r["symbol"] for r in ds1["rows"]] == ["AAPL", "MSFT"]
