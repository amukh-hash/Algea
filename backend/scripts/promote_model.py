#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.ml_platform.config import MLPlatformConfig
from backend.app.ml_platform.registry.promotion import promote_if_eligible
from backend.app.ml_platform.registry.store import ModelRegistryStore


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--to", default="prod")
    ap.add_argument("--metrics", required=True, help="json metrics string")
    args = ap.parse_args()

    cfg = MLPlatformConfig()
    store = ModelRegistryStore(cfg.registry_db_path, cfg.model_root)
    metrics = json.loads(args.metrics)
    ok = promote_if_eligible(store, args.model, args.version, metrics, to_alias=args.to)
    if not ok:
        raise SystemExit("promotion gate failed")
    print(f"promoted {args.model}:{args.version} -> {args.to}")


if __name__ == "__main__":
    main()
