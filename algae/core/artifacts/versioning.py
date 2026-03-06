from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from typing import Any, Dict


def stable_hash(payload: Dict[str, Any]) -> str:
    dumped = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(dumped).hexdigest()[:12]


def dataclass_hash(obj: Any) -> str:
    return stable_hash(asdict(obj))
