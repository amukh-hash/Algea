import json
from pathlib import Path

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.models.vol_surface.loader import VolSurfaceLoader
from backend.app.ml_platform.models.vol_surface.service import VolSurfaceService
from backend.app.ml_platform.models.vol_surface.types import VolSurfaceRequest
from backend.app.ml_platform.registry.store import ModelRegistryStore


def test_vol_surface_trace_emission(tmp_path: Path):
    cfg = MLPlatformConfig(registry_db_path=tmp_path / "r.sqlite", model_root=tmp_path / "models", trace_root=tmp_path / "traces")
    store = ModelRegistryStore(cfg.registry_db_path, cfg.model_root)
    store.publish_version("vol_surface", "v1", "x", {"pinball_loss": 0.1, "calibration_score": 0.8, "edge_hit_rate": 0.6}, {"hidden_size": 8}, {"feature_schema": {}, "drift_baseline": {"7": {"rv_hist_20_mean": 0.2, "rv_hist_20_std": 1.0}}, "calibration": {"calibration_score": 0.8}})
    store.set_alias("vol_surface", "prod", "v1")
    svc = VolSurfaceService(VolSurfaceLoader(store), cfg.trace_root)
    svc.forecast(VolSurfaceRequest(asof="2026-01-01", underlying_symbol="SPY", tenors=[7], history={7: [{"rv_hist_20": 0.2}] * 5}, trace_id="vs-tr"))
    payload = json.loads((cfg.trace_root / "vs-tr.json").read_text(encoding="utf-8"))
    assert "uncertainty" in payload and "drift_score" in payload
