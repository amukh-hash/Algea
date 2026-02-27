from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .chain_schema import validate_chain_rows


def snapshot_id(underlying_symbol: str, asof: str, chain_params_hash: str) -> str:
    return hashlib.sha256(f"{underlying_symbol}|{asof}|{chain_params_hash}".encode("utf-8")).hexdigest()


def write_snapshot_atomic(root: Path, rows: list[dict], snap_id: str) -> Path:
    validate_chain_rows(rows)
    root.mkdir(parents=True, exist_ok=True)
    tmp = root / f"{snap_id}.tmp.json"
    dest = root / f"{snap_id}.json"
    tmp.write_text(json.dumps(rows, sort_keys=True), encoding="utf-8")
    tmp.replace(dest)
    return dest
