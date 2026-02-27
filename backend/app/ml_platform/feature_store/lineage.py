from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def build_lineage_manifest(
    dataset_id: str,
    source_ranges: list[str],
    input_files: Iterable[Path],
    feature_code_hash: str,
    universe_definition_hash: str,
) -> dict:
    files = sorted((str(p), _sha(p)) for p in input_files)
    payload = {
        "dataset_id": dataset_id,
        "source_ranges": source_ranges,
        "input_file_hashes": files,
        "feature_code_version_hash": feature_code_hash,
        "universe_definition_hash": universe_definition_hash,
    }
    payload["lineage_hash"] = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    return payload
