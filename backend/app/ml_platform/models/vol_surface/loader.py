from __future__ import annotations

from ...registry.store import ModelRegistryStore
from .artifact import load_vol_surface_artifact


class VolSurfaceLoader:
    def __init__(self, store: ModelRegistryStore):
        self.store = store

    def load_alias(self, alias: str = "prod") -> dict:
        version = self.store.resolve_alias("vol_surface", alias)
        if not version:
            raise RuntimeError(f"vol_surface alias '{alias}' is not set")
        root = self.store.model_root / "vol_surface" / version
        payload = load_vol_surface_artifact(root)
        payload["model_version"] = version
        return payload
