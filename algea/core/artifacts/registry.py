from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable


@dataclass(frozen=True)
class ArtifactRecord:
    name: str
    path: Path
    version: str


class ArtifactRegistry:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.records: Dict[str, ArtifactRecord] = {}

    def register(self, name: str, path: Path, version: str) -> None:
        self.records[name] = ArtifactRecord(name=name, path=path, version=version)

    def dump(self, destination: Path) -> None:
        payload = {
            name: {"path": str(record.path), "version": record.version}
            for name, record in self.records.items()
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def list_records(self) -> Iterable[ArtifactRecord]:
        return self.records.values()
