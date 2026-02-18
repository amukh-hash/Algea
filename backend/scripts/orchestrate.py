from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.orchestrator.calendar import Session
from backend.app.orchestrator.config import OrchestratorConfig
from backend.app.orchestrator.orchestrator import Orchestrator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Session-aware trading orchestrator")
    parser.add_argument("--once", action="store_true", help="Run one tick and exit")
    parser.add_argument("--daemon", action="store_true", help="Run forever")
    parser.add_argument("--asof", type=lambda s: date.fromisoformat(s))
    parser.add_argument("--session", choices=[s.value for s in Session])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--poll-interval", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_once = args.once or not args.daemon

    config = OrchestratorConfig()
    if args.poll_interval is not None:
        config.poll_interval_s = args.poll_interval

    orch = Orchestrator(config=config)
    forced_session = Session(args.session) if args.session else None

    if run_once:
        with orch.locks.acquire_global_lock():
            result = orch.run_once(asof=args.asof, forced_session=forced_session, dry_run=args.dry_run)
            print(json.dumps(result.__dict__, indent=2))
        return

    with orch.locks.acquire_global_lock():
        while True:
            result = orch.run_once(asof=args.asof, forced_session=forced_session, dry_run=args.dry_run)
            print(json.dumps(result.__dict__, indent=2))
            time.sleep(config.poll_interval_s)


if __name__ == "__main__":
    main()
