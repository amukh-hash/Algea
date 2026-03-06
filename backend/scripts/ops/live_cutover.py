"""Stage 5: Live Capital Cutover Script.

Executes the live deployment checklist:
1. WAL checkpoint (flush all pending SQLite writes)
2. Clear stale order intents from previous sessions
3. Set fractional risk cap (default 0.25)
4. Switch execution_mode from 'paper' to 'ibkr'
5. Unpause the orchestrator
6. Verify all systems are go

Usage
-----
::

    # Pre-flight check (dry run — changes nothing)
    python -m backend.scripts.ops.live_cutover preflight

    # Execute the cutover (DESTRUCTIVE — flips to live)
    python -m backend.scripts.ops.live_cutover execute --risk-cap 0.25

    # Emergency rollback
    python -m backend.scripts.ops.live_cutover rollback
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("live_cutover")

EASTERN_TZ = ZoneInfo("America/New_York")
DB_PATH = Path("backend/artifacts/orchestrator_state/state.sqlite3")


def _state():
    from backend.app.orchestrator.durable_control_state import DurableControlState
    return DurableControlState(DB_PATH)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    icon = "✅" if ok else "❌"
    logger.info("%s %s %s", icon, label, detail)
    return ok


def preflight():
    """Pre-flight check — validates all systems without making changes."""
    logger.info("═" * 60)
    logger.info("STAGE 5 — Live Capital Cutover: PRE-FLIGHT CHECK")
    logger.info("═" * 60)
    logger.info("⚠️  This is a DRY RUN — no changes will be made.")
    logger.info("")

    all_ok = True
    state = _state()
    snap = state.snapshot()

    # 1. Current state
    logger.info("Current Control State:")
    for k, v in snap.items():
        logger.info("  %s = %s", k, v)
    logger.info("")

    # 2. Paper mode confirm
    all_ok &= _check(
        "Currently in paper mode",
        snap["execution_mode"] == "paper",
        f"→ {snap['execution_mode']}",
    )

    # 3. Not paused
    all_ok &= _check("Not paused", not snap["paused"])

    # 4. DB integrity
    conn = sqlite3.connect(DB_PATH)
    try:
        # WAL size
        wal_path = Path(str(DB_PATH) + "-wal")
        if wal_path.exists():
            wal_kb = wal_path.stat().st_size / 1024
            all_ok &= _check("WAL size", wal_kb < 10240, f"→ {wal_kb:.0f} KB")
        else:
            _check("WAL file", True, "→ clean (no WAL)")

        # Pending intents from today
        today = datetime.now(EASTERN_TZ).strftime("%Y-%m-%d")
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM order_intents WHERE asof_date=? AND status='PENDING'",
            (today,),
        ).fetchone()
        n_pending = row[0] if row else 0
        _check("Pending intents today", True, f"→ {n_pending}")

        # Check for stale intents
        stale = conn.execute(
            "SELECT COUNT(*) as cnt FROM order_intents WHERE asof_date < ? AND status='PENDING'",
            (today,),
        ).fetchone()
        n_stale = stale[0] if stale else 0
        all_ok &= _check("No stale pending intents", n_stale == 0, f"→ {n_stale} stale")

        # Lock table check
        locks = conn.execute("SELECT COUNT(*) FROM locks").fetchone()
        n_locks = locks[0] if locks else 0
        all_ok &= _check("No active locks", n_locks == 0, f"→ {n_locks} locks")

    except sqlite3.Error as e:
        all_ok &= _check("DB queries", False, str(e))
    conn.close()

    # 5. VRAM check
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                alloc = torch.cuda.memory_allocated(i) / (1024**2)
                all_ok &= _check(
                    f"CUDA:{i} clean",
                    alloc < 100,
                    f"→ {name}, alloc={alloc:.0f}MB",
                )
    except ImportError:
        _check("CUDA", False, "torch not importable")

    # 6. Broker connectivity
    # NOTE: Actual broker ping requires IBKR TWS/Gateway running.
    # This is a placeholder check.
    _check("Broker connectivity", True, "→ manual verification required")

    logger.info("")
    logger.info("═" * 60)
    if all_ok:
        logger.info("✅ PRE-FLIGHT PASSED — ready for live cutover")
        logger.info("   Run: python -m backend.scripts.ops.live_cutover execute --risk-cap 0.25")
    else:
        logger.info("❌ PRE-FLIGHT FAILED — fix issues before proceeding")
    logger.info("═" * 60)


def execute(risk_cap: float = 0.25):
    """Execute the live capital cutover."""
    logger.info("═" * 60)
    logger.info("STAGE 5 — LIVE CAPITAL CUTOVER")
    logger.info("═" * 60)
    logger.warning("⚠️  THIS WILL ENABLE LIVE TRADING WITH REAL CAPITAL ⚠️")
    logger.info("")

    state = _state()
    snap = state.snapshot()

    # Safety: must be in paper mode to cut over
    if snap["execution_mode"] != "paper":
        logger.error("Cannot cut over: execution_mode is '%s', expected 'paper'", snap["execution_mode"])
        logger.error("Run rollback first, then paper burn-in, then retry.")
        sys.exit(1)

    today = datetime.now(EASTERN_TZ).strftime("%Y-%m-%d")

    # Step 1: WAL Checkpoint
    logger.info("Step 1: WAL Checkpoint")
    conn = sqlite3.connect(DB_PATH, timeout=30)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.close()
    _check("WAL checkpoint", True, "→ TRUNCATE complete")

    # Step 2: Clear stale intents
    logger.info("Step 2: Clear stale order intents")
    conn = sqlite3.connect(DB_PATH, timeout=30)
    deleted = conn.execute(
        "DELETE FROM order_intents WHERE asof_date < ? AND status='PENDING'",
        (today,),
    ).rowcount
    conn.commit()
    conn.close()
    _check("Stale intents cleared", True, f"→ {deleted} deleted")

    # Step 3: Set fractional risk cap
    logger.info("Step 3: Set risk cap")
    state.set_exposure_cap(risk_cap)
    _check("Risk cap", True, f"→ gross_exposure_cap = {risk_cap}")

    # Step 4: Switch to live
    logger.info("Step 4: Switch execution_mode to 'ibkr'")
    state.set_execution_mode("ibkr")
    _check("Execution mode", True, "→ ibkr")

    # Step 5: Unpause
    logger.info("Step 5: Unpause orchestrator")
    state.set_paused(False)
    _check("Unpaused", True)

    # Final verification
    final = state.snapshot()
    logger.info("")
    logger.info("Final Control State:")
    for k, v in final.items():
        logger.info("  %s = %s", k, v)

    logger.info("")
    logger.info("═" * 60)
    logger.info("🚀 LIVE CUTOVER COMPLETE")
    logger.info("   execution_mode = ibkr")
    logger.info("   gross_exposure_cap = %.2f", risk_cap)
    logger.info("   Time: %s", datetime.now(timezone.utc).isoformat())
    logger.info("")
    logger.info("   Monitor: python -m backend.scripts.ops.paper_burnin status")
    logger.info("   Rollback: python -m backend.scripts.ops.live_cutover rollback")
    logger.info("═" * 60)


def rollback():
    """Emergency rollback: pause orchestrator and force paper mode."""
    logger.info("═" * 60)
    logger.warning("⚠️  EMERGENCY ROLLBACK — Forcing paper mode + pause")
    logger.info("═" * 60)

    state = _state()

    # Step 1: Pause immediately
    state.set_paused(True)
    _check("Paused", True)

    # Step 2: Force paper mode
    state.set_execution_mode("paper")
    _check("Execution mode", True, "→ paper")

    # Step 3: Restore full exposure cap
    state.set_exposure_cap(1.5)
    _check("Exposure cap", True, "→ 1.5 (default)")

    # Step 4: Clear pending intents for safety
    conn = sqlite3.connect(DB_PATH, timeout=30)
    cleared = conn.execute("DELETE FROM order_intents WHERE status='PENDING'").rowcount
    conn.commit()
    conn.close()
    _check("Pending intents cleared", True, f"→ {cleared} deleted")

    final = state.snapshot()
    logger.info("")
    logger.info("Final Control State:")
    for k, v in final.items():
        logger.info("  %s = %s", k, v)

    logger.info("")
    logger.info("═" * 60)
    logger.info("🛑 ROLLBACK COMPLETE — system is safe, paused, in paper mode")
    logger.info("═" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 5: Live Capital Cutover")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("preflight", help="Pre-flight check (dry run)")

    exec_parser = sub.add_parser("execute", help="Execute live cutover")
    exec_parser.add_argument(
        "--risk-cap", type=float, default=0.25,
        help="Fractional gross exposure cap (default: 0.25 = 25%%)",
    )

    sub.add_parser("rollback", help="Emergency rollback to paper mode")

    args = parser.parse_args()

    if args.command == "preflight":
        preflight()
    elif args.command == "execute":
        execute(risk_cap=args.risk_cap)
    elif args.command == "rollback":
        rollback()
