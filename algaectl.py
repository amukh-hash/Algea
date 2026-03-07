#!/usr/bin/env python3
"""algaectl — Canonical CLI for the Algae trading system.

Usage::

    python algaectl.py run orchestrator --mode paper
    python algaectl.py run backtest --strategy vrp
    python algaectl.py ops cutover-check
    python algaectl.py ops sweep-runs --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s",
)
logger = logging.getLogger("algaectl")


def _cmd_run_orchestrator(args: argparse.Namespace) -> int:
    """Start the orchestrator in the specified mode."""
    from backend.app.core.runtime_mode import RuntimeMode

    mode_map = {
        "live": RuntimeMode.LIVE,
        "paper": RuntimeMode.PAPER,
        "noop": RuntimeMode.NOOP,
        "stub": RuntimeMode.STUB,
    }
    mode = mode_map.get(args.mode)
    if mode is None:
        logger.error("Unknown mode: %s (valid: %s)", args.mode, list(mode_map.keys()))
        return 1

    logger.info("Starting orchestrator in mode=%s", mode.value)

    import os
    os.environ["ALGAE_RUNTIME_MODE"] = mode.value

    from scripts.run_orchestrator_cycle import main as run_orch
    run_orch()
    return 0


def _cmd_run_backtest(args: argparse.Namespace) -> int:
    """Run a backtest for the given strategy."""
    logger.info("Running backtest for strategy=%s", args.strategy)
    from backend.scripts.research.run_backtest import main as run_bt
    run_bt()
    return 0


def _cmd_ops_cutover(args: argparse.Namespace) -> int:
    """Run the live cutover preflight checklist."""
    logger.info("Running cutover preflight checks...")
    from backend.scripts.ops.live_cutover import run_cutover_check
    ok = run_cutover_check()
    return 0 if ok else 1


def _cmd_ops_sweep(args: argparse.Namespace) -> int:
    """Sweep stale run artifacts."""
    import shutil

    runs_dir = Path("runs")
    if not runs_dir.exists():
        logger.info("No runs/ directory to sweep")
        return 0

    dirs = sorted(runs_dir.iterdir())
    logger.info("Found %d run directories", len(dirs))
    if args.dry_run:
        for d in dirs:
            logger.info("  [DRY-RUN] would remove: %s", d)
        return 0

    for d in dirs:
        if d.is_dir():
            shutil.rmtree(d)
            logger.info("  removed: %s", d)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="algaectl",
        description="Canonical CLI for the Algae trading system.",
    )
    sub = parser.add_subparsers(dest="command", help="Top-level command")

    # --- run ---
    run_parser = sub.add_parser("run", help="Run a system component")
    run_sub = run_parser.add_subparsers(dest="run_command")

    orch = run_sub.add_parser("orchestrator", help="Start the orchestrator")
    orch.add_argument("--mode", required=True, choices=["live", "paper", "noop", "stub"])
    orch.set_defaults(func=_cmd_run_orchestrator)

    bt = run_sub.add_parser("backtest", help="Run a strategy backtest")
    bt.add_argument("--strategy", required=True, choices=["vrp", "statarb", "core"])
    bt.set_defaults(func=_cmd_run_backtest)

    # --- ops ---
    ops_parser = sub.add_parser("ops", help="Operational commands")
    ops_sub = ops_parser.add_subparsers(dest="ops_command")

    co = ops_sub.add_parser("cutover-check", help="Run live cutover preflight")
    co.set_defaults(func=_cmd_ops_cutover)

    sweep = ops_sub.add_parser("sweep-runs", help="Clean up stale run artifacts")
    sweep.add_argument("--dry-run", action="store_true", help="Show what would be deleted")
    sweep.set_defaults(func=_cmd_ops_sweep)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
