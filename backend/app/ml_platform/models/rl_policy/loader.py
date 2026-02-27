from __future__ import annotations

from .artifact import load_rl_policy_artifact
from ...registry.store import ModelRegistryStore


class RLPolicyLoader:
    def __init__(self, store: ModelRegistryStore):
        self.store = store

    def load_alias(self, alias: str = "prod") -> dict:
        version = self.store.resolve_alias("rl_policy", alias)
        if not version:
            raise RuntimeError(f"rl_policy alias '{alias}' is not set")
        root = self.store.model_root / "rl_policy" / version
        bundle = load_rl_policy_artifact(root)
        return {"model_version": version, **bundle}
