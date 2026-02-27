from __future__ import annotations

import json
from pathlib import Path

from .hashes import hash_payload


def replay_session(trace_paths: list[Path]) -> dict:
    decisions: list[dict] = []
    for p in sorted(trace_paths):
        decisions.append(json.loads(p.read_text(encoding="utf-8")))
    decision_hash = hash_payload({"decisions": decisions})
    return {"count": len(decisions), "decision_hash": decision_hash}
