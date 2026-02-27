from __future__ import annotations

import json
from pathlib import Path

from ...registry.store import ModelRegistryStore
from .types import Chronos2ModelBundle


class Chronos2Loader:
    def __init__(self, store: ModelRegistryStore):
        self.store = store

    def load_alias(self, alias: str = "prod") -> Chronos2ModelBundle:
        version = self.store.resolve_alias("chronos2", alias)
        if not version:
            raise RuntimeError(f"chronos2 alias '{alias}' is not set")
        root = self.store.model_root / "chronos2" / version
        cfg = json.loads((root / "model_config.json").read_text(encoding="utf-8"))
        calibration = json.loads((root / "calibration.json").read_text(encoding="utf-8"))
        drift = json.loads((root / "drift_baseline.json").read_text(encoding="utf-8"))
        return Chronos2ModelBundle(
            model_version=version,
            config=cfg,
            calibration=calibration,
            drift_baseline=drift,
            artifact_dir=str(root),
        )
