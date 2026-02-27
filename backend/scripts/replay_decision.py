#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace_id", required=True)
    ap.add_argument("--trace_root", default="backend/artifacts/traces")
    args = ap.parse_args()

    trace_path = Path(args.trace_root) / f"{args.trace_id}.json"
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    print(json.dumps({"trace_id": args.trace_id, "replay": "ok", "payload": payload}, indent=2))


if __name__ == "__main__":
    main()
