from __future__ import annotations

import json

from ...registry.store import ModelRegistryStore


class SMoELoader:
    def __init__(self, store: ModelRegistryStore):
        self.store = store

    def load_alias(self, alias: str = "prod") -> dict:
        version = self.store.resolve_alias("selector_smoe", alias)
        if not version:
            raise RuntimeError(f"selector_smoe alias '{alias}' is not set")
        root = self.store.model_root / "selector_smoe" / version
        return {
            "model_version": version,
            "config": json.loads((root / "model_config.json").read_text(encoding="utf-8")),
            "drift_baseline": json.loads((root / "drift_baseline.json").read_text(encoding="utf-8")),
            "calibration": json.loads((root / "calibration.json").read_text(encoding="utf-8")),
        }
