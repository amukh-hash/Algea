"""
Orchestrator One-Shot Launcher

Fires a single Orchestrator cycle (all enabled sleeves) and terminates.
Designed for Windows Task Scheduler: process-level teardown guarantees
zero VRAM leaks across multi-year deployment.

Usage:
    python scripts/run_orchestrator_cycle.py              # Auto-detect session
    python scripts/run_orchestrator_cycle.py --premarket  # Force PREMARKET
    python scripts/run_orchestrator_cycle.py --eod        # Force EOD flatten
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import date, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("PYTHONPATH", str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/orchestrator.log", mode="a"),
    ],
)
logger = logging.getLogger("OrchestratorLauncher")

# Load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def run_cycle(forced_session: str | None = None):
    """Fire a single Orchestrator cycle."""
    from backend.app.orchestrator.orchestrator import Orchestrator
    from backend.app.orchestrator.calendar import Session

    session_map = {
        "premarket": Session.PREMARKET,
        "intraday": Session.INTRADAY,
        "eod": Session.PRECLOSE,
        "close": Session.CLOSE,
    }

    session = session_map.get(forced_session) if forced_session else None

    logger.info("=" * 60)
    logger.info("ORCHESTRATOR CYCLE START")
    logger.info("  Time: %s", datetime.now().isoformat())
    logger.info("  Forced session: %s", forced_session or "AUTO")
    logger.info("  Sleeves: COOC=%s Chronos2=%s SMoE=%s VRP=%s StatArb=%s",
                os.getenv("ENABLE_CORE_COOC_SLEEVE", "?"),
                os.getenv("ENABLE_CHRONOS2_SLEEVE", "?"),
                os.getenv("ENABLE_SMOE_SELECTOR", "?"),
                os.getenv("ENABLE_VOL_SURFACE_VRP", "?"),
                os.getenv("ENABLE_STATARB_SLEEVE", "?"))
    logger.info("=" * 60)

    orch = Orchestrator()
    result = orch.run_once(forced_session=session)

    logger.info("=" * 60)
    logger.info("ORCHESTRATOR CYCLE COMPLETE")
    logger.info("  Run ID: %s", result.run_id)
    logger.info("  Session: %s", result.session)
    logger.info("  Ran:     %s", result.ran_jobs)
    logger.info("  Skipped: %s", result.skipped_jobs)
    logger.info("  Failed:  %s", result.failed_jobs)
    logger.info("=" * 60)

    # Write shadow ledger entry
    shadow_dir = Path("backend/artifacts/orchestrator/shadow_ledger")
    shadow_dir.mkdir(parents=True, exist_ok=True)
    shadow_path = shadow_dir / f"shadow_{date.today().strftime('%Y%m%d')}.jsonl"
    entry = {
        "timestamp": datetime.now().isoformat(),
        "run_id": result.run_id,
        "session": result.session,
        "ran_jobs": result.ran_jobs,
        "skipped_jobs": result.skipped_jobs,
        "failed_jobs": result.failed_jobs,
    }
    with open(shadow_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    logger.info("Shadow ledger: %s", shadow_path)

    return result


def main():
    parser = argparse.ArgumentParser(description="Orchestrator One-Shot Cycle")
    parser.add_argument("--premarket", action="store_true", help="Force PREMARKET session")
    parser.add_argument("--intraday", action="store_true", help="Force INTRADAY session")
    parser.add_argument("--eod", action="store_true", help="Force POSTMARKET/EOD session")
    args = parser.parse_args()

    if args.premarket:
        session = "premarket"
    elif args.intraday:
        session = "intraday"
    elif args.eod:
        session = "eod"
    else:
        session = None

    run_cycle(session)


if __name__ == "__main__":
    main()
