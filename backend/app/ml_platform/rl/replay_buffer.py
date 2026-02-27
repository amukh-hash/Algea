from __future__ import annotations

import json
from pathlib import Path


class ReplayBuffer:
    def __init__(self, path: Path, schema_version: int = 1):
        self.path = path
        self.schema_version = schema_version
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, transitions: list[dict]) -> None:
        payload = self.load()
        payload["transitions"].extend(transitions)
        self.path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    def load(self) -> dict:
        if not self.path.exists():
            return {"schema_version": self.schema_version, "transitions": []}
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != self.schema_version:
            raise ValueError("replay buffer schema mismatch")
        if not isinstance(payload.get("transitions", []), list):
            raise ValueError("replay buffer integrity failure")
        return payload
