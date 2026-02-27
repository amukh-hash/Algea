from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..iv_grid import IVGridSpec, ordered_grid_keys


def build_vol_surface_grid_dataset(
    rows: list[dict],
    out_root: Path,
    spec: IVGridSpec | None = None,
) -> dict:
    spec = spec or IVGridSpec()
    keys = ordered_grid_keys(spec)
    payload = []
    for r in sorted(rows, key=lambda x: x["ts"]):
        feat = [[float(r.get("iv", {}).get(k, 0.0)), float(r.get("liq", {}).get(k, 0.0)), float(r.get("ret", {}).get(k, 0.0))] for k in keys]
        mask = [1.0 if k in r.get("iv", {}) else 0.0 for k in keys]
        target = [float(r.get("target", {}).get(k, 0.0)) for k in keys]
        payload.append({"ts": r["ts"], "features": feat, "mask": mask, "target": target})
    manifest = {"rows": len(payload), "keys": keys, "tenors": list(spec.tenors), "buckets": list(spec.buckets)}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    out_dir = out_root / digest
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "dataset.json").write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    manifest["dataset_hash"] = digest
    (out_dir / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2), encoding="utf-8")
    return {"dataset_id": digest, "path": out_dir, "manifest": manifest}
