"""Operator-facing status CLI for the orchestrator.

Usage:
    python backend/scripts/orch_status.py
    python backend/scripts/orch_status.py --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.orchestrator.config import OrchestratorConfig
from backend.app.orchestrator.health import summarize_health
from backend.app.orchestrator.state_store import StateStore


def _fmt_age(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.0f}s ago"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def _fmt_notional(value: float | int | None) -> str:
    if value is None:
        return "—"
    return f"${value:,.0f}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Orchestrator status")
    parser.add_argument("--json", action="store_true", help="Output raw JSON")
    args = parser.parse_args()

    config = OrchestratorConfig()
    state = StateStore(config.db_path)
    health = summarize_health(config.artifact_root, state)

    if args.json:
        print(json.dumps(health, indent=2, default=str))
        return

    hb = health.get("heartbeat")
    session = hb.get("session", "unknown") if hb else "unknown"
    mode = hb.get("mode", "unknown") if hb else "unknown"
    age = _fmt_age(health.get("heartbeat_age_s"))

    last_run = health.get("last_success")
    last_run_status = "none"
    last_run_id = "—"
    if last_run:
        last_run_status = last_run.get("status", "unknown")
        last_run_id = last_run.get("run_id", "—")[:12]

    last_fail_job = health.get("last_job_failure")
    fail_line = "none"
    if last_fail_job:
        fail_line = f"{last_fail_job['job_name']} – \"{last_fail_job.get('error_summary', 'unknown')}\""
        if last_fail_job.get("ended_at"):
            fail_line += f" ({last_fail_job['ended_at'][:19]})"

    routed = health.get("last_routed")
    routed_line = "none"
    if routed:
        count = routed.get("order_count", 0)
        notional = _fmt_notional(routed.get("total_abs_notional"))
        routed_line = f"{count} orders, {notional} notional"
        if routed.get("dry_run"):
            routed_line += " (dry-run)"

    print(f"Session:          {session}")
    print(f"Mode:             {mode}")
    print(f"Heartbeat age:    {age}")
    print(f"Last run:         {last_run_status} (run_id {last_run_id})")
    print(f"Last failure:     {fail_line}")
    print(f"Last routed:      {routed_line}")


if __name__ == "__main__":
    main()
