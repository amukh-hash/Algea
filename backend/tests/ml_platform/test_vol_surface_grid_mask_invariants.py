import json

from backend.app.ml_platform.feature_store.builders.vol_surface_grid import build_vol_surface_grid_dataset


def test_vol_surface_grid_mask_invariants(tmp_path):
    rows = [{"ts": "2026-01-01T00:00:00", "iv": {"7:ATM": 0.2}, "target": {"7:ATM": 0.2}}]
    out = build_vol_surface_grid_dataset(rows, tmp_path)
    data = json.loads((out["path"] / "dataset.json").read_text(encoding="utf-8"))
    assert set(data[0]["mask"]).issubset({0.0, 1.0})
