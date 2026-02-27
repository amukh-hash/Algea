from __future__ import annotations

import json

from ...registry.store import ModelRegistryStore


class ITransformerLoader:
    def __init__(self, store: ModelRegistryStore):
        self.store = store

    def load_alias(self, alias: str = "prod") -> dict:
        version = self.store.resolve_alias("itransformer", alias)
        if not version:
            raise RuntimeError(f"itransformer alias '{alias}' is not set")
        root = self.store.model_root / "itransformer" / version
        return {
            "model_version": version,
            "config": json.loads((root / "model_config.json").read_text(encoding="utf-8")),
            "drift_baseline": json.loads((root / "drift_baseline.json").read_text(encoding="utf-8")),
        }
