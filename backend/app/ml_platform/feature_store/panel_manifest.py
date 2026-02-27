from __future__ import annotations

import hashlib
import json


def build_panel_manifest(payload: dict) -> dict:
    out = dict(payload)
    out["manifest_hash"] = hashlib.sha256(json.dumps(out, sort_keys=True).encode("utf-8")).hexdigest()
    return out
