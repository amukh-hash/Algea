from __future__ import annotations

import json
from pathlib import Path


class VolSurfaceGridLoader:
    def __init__(self, store):
        self.store = store

    def load_alias(self, alias: str = "prod") -> dict:
        version = self.store.resolve_alias("vol_surface_grid", alias)
        if not version:
            raise RuntimeError(f"vol_surface_grid alias '{alias}' is not set")
        path = self.store.model_root / "vol_surface_grid" / version
        return {
            "model_version": version,
            "config": json.loads((path / "model_config.json").read_text(encoding="utf-8")),
        }
