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
    parser.add_argument(
        "--broker",
        choices=["stub", "ibkr"],
        default="stub",
        help="Broker backend: 'stub' (default, no real orders) or 'ibkr' (connects to IBKR TWS/Gateway)",
    )
    return parser.parse_args()


def _build_broker(broker_type: str):
    """Create the appropriate broker adapter."""
    if broker_type == "ibkr":
        from backend.app.orchestrator.broker_ibkr_adapter import IBKRBrokerAdapter

        print("[orchestrate] Using IBKR live broker adapter (from env vars)")
        return IBKRBrokerAdapter.from_env()
    else:
        from backend.app.orchestrator.broker import PaperBrokerStub

        print("[orchestrate] Using paper broker stub")
        return PaperBrokerStub()


def main() -> None:
    args = parse_args()
    run_once = args.once or not args.daemon

    config = OrchestratorConfig()
    if args.poll_interval is not None:
        config.poll_interval_s = args.poll_interval

    broker = _build_broker(args.broker)
    orch = Orchestrator(config=config, broker=broker)
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
