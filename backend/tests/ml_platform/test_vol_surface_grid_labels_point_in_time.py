from backend.app.ml_platform.feature_store.builders.vol_surface_grid import build_vol_surface_grid_dataset


def test_vol_surface_grid_labels_point_in_time(tmp_path):
    rows = [
        {"ts": "2026-01-01T00:00:00", "iv": {"7:ATM": 0.2}, "target": {"7:ATM": 0.25}},
        {"ts": "2026-01-02T00:00:00", "iv": {"7:ATM": 0.21}, "target": {"7:ATM": 0.26}},
    ]
    out = build_vol_surface_grid_dataset(rows, tmp_path)
    data = (out["path"] / "dataset.json").read_text(encoding="utf-8")
    assert "2026-01-01" in data and "2026-01-02" in data
