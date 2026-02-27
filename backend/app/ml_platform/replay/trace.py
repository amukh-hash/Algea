from __future__ import annotations

import json
from pathlib import Path


def write_trace(trace_root: Path, trace_id: str, payload: dict) -> Path:
    path = trace_root / f"{trace_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return path
