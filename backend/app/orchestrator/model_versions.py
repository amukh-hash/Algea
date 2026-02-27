from __future__ import annotations

import json
from pathlib import Path


MODEL_KEYS = ["chronos2", "selector_smoe", "vol_surface", "vol_surface_grid", "itransformer", "rl_policy"]


def record_model_versions(
    path: Path,
    versions: dict[str, str | dict],
    *,
    run_id: str = "",
    asof_date: str = "",
    session: str = "",
) -> Path:
    payload = {
        "run_id": run_id,
        "asof_date": asof_date,
        "session": session,
        "models": {},
    }
    for key in MODEL_KEYS:
        value = versions.get(key, "unknown")
        if isinstance(value, dict):
            entry = {
                "model_name": str(value.get("model_name", key)),
                "model_alias": str(value.get("model_alias", "")),
                "model_version": str(value.get("model_version", "unknown")),
                "endpoint_name": str(value.get("endpoint_name", "")),
            }
            if "latency_ms" in value:
                entry["latency_ms"] = float(value["latency_ms"])
        else:
            entry = {
                "model_name": key,
                "model_alias": "",
                "model_version": str(value),
                "endpoint_name": "",
            }
        payload["models"][key] = entry
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    return path
