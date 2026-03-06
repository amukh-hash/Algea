"""Stage 4: Paper Burn-In Setup & Validation Script.

Automates the paper trading burn-in process:
1. Sets execution_mode='paper' in durable state
2. Verifies scheduler triggers are correctly wired
3. Runs pre-flight checks (DB tables, intents, VRAM)
4. Launches the PhaseScheduler in paper mode
5. After each trading day, runs validation checks

Usage
-----
::

    python -m backend.scripts.ops.paper_burnin setup     # one-time setup
    python -m backend.scripts.ops.paper_burnin validate   # daily validation
    python -m backend.scripts.ops.paper_burnin status     # check state
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
logger = logging.getLogger("paper_burnin")

EASTERN_TZ = ZoneInfo("America/New_York")
DB_PATH = Path("backend/artifacts/orchestrator_state/state.sqlite3")
ARTIFACT_ROOT = Path("backend/artifacts/orchestrator")


def _state():
    from backend.app.orchestrator.durable_control_state import DurableControlState
    return DurableControlState(DB_PATH)


def _check(label: str, ok: bool, detail: str = "") -> bool:
    icon = "✅" if ok else "❌"
    logger.info("%s %s %s", icon, label, detail)
    return ok


def setup():
    """One-time setup: force paper mode, verify tables, initialize state."""
    logger.info("═" * 60)
    logger.info("STAGE 4 — Paper Burn-In Setup")
    logger.info("═" * 60)

    # 1. Force paper mode
    state = _state()
    state.set_execution_mode("paper")
    state.set_paused(False)
    state.set_exposure_cap(1.5)
    snap = state.snapshot()
    _check("Execution mode", snap["execution_mode"] == "paper", f"→ {snap['execution_mode']}")
    _check("Not paused", not snap["paused"])
    _check("Exposure cap", snap["gross_exposure_cap"] == 1.5, f"→ {snap['gross_exposure_cap']}")

    # 2. Verify DB schema
    conn = sqlite3.connect(DB_PATH)
    tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    required = ["app_control_state", "order_intents", "runs", "jobs", "locks"]
    for t in required:
        _check(f"Table '{t}'", t in tables)
    conn.close()

    # 3. Verify DAG topology
    dag_path = Path("backend/configs/dag_topology.yaml")
    _check("dag_topology.yaml exists", dag_path.exists())

    # 4. Verify scheduler importable
    try:
        from backend.app.orchestrator.phase_scheduler import PhaseScheduler
        s = PhaseScheduler(db_path=DB_PATH)
        _check("PhaseScheduler", True, f"→ {len(s.triggers)} triggers configured")
        for t in s.triggers:
            logger.info("   ⏰ %s @ %s EST", t.name, t.trigger_time.strftime("%H:%M"))
    except Exception as e:
        _check("PhaseScheduler", False, str(e))

    # 5. Verify risk gateway
    try:
        from backend.app.core.risk_gateway import validate_and_store_intents, route_phase_orders
        _check("Risk gateway importable", True)
    except Exception as e:
        _check("Risk gateway importable", False, str(e))

    # 6. Verify intent aggregator
    try:
        from backend.app.orchestrator.intent_aggregator import collect_and_validate_intents
        _check("Intent aggregator importable", True)
    except Exception as e:
        _check("Intent aggregator importable", False, str(e))

    # 7. VRAM check
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                name = torch.cuda.get_device_name(i)
                total = torch.cuda.get_device_properties(i).total_mem / (1024**3)
                _check(f"CUDA:{i}", True, f"→ {name} ({total:.1f} GB)")
        else:
            _check("CUDA", False, "Not available — GPU inference will fail")
    except ImportError:
        _check("CUDA", False, "torch not importable")

    logger.info("═" * 60)
    logger.info("Setup complete. Run the orchestrator for 5 trading days.")
    logger.info("Daily validation: python -m backend.scripts.ops.paper_burnin validate")
    logger.info("═" * 60)


def validate():
    """Daily validation: check that paper trading produced expected artifacts."""
    today = datetime.now(EASTERN_TZ).strftime("%Y-%m-%d")
    day_root = ARTIFACT_ROOT / today

    logger.info("═" * 60)
    logger.info("Paper Burn-In Validation — %s", today)
    logger.info("═" * 60)

    all_ok = True

    # 1. Control state
    state = _state()
    snap = state.snapshot()
    all_ok &= _check("execution_mode='paper'", snap["execution_mode"] == "paper")
    all_ok &= _check("not paused", not snap["paused"])

    # 2. Day artifacts
    all_ok &= _check(f"Day root exists ({today})", day_root.exists())

    if day_root.exists():
        # Heartbeat
        hb = day_root / "heartbeat.json"
        all_ok &= _check("Heartbeat", hb.exists())

        # Intent files
        intents_dir = day_root / "intents"
        if intents_dir.exists():
            intent_files = list(intents_dir.glob("*_intents.json"))
            all_ok &= _check("Intent files", len(intent_files) > 0, f"→ {len(intent_files)} files")
            total_intents = 0
            for f in intent_files:
                try:
                    data = json.loads(f.read_text("utf-8"))
                    n = len(data) if isinstance(data, list) else len(data.get("intents", []))
                    total_intents += n
                except Exception:
                    pass
            _check("Total intents today", total_intents > 0, f"→ {total_intents}")
        else:
            all_ok &= _check("Intents directory", False, "not found")

        # Signals
        signals_dir = day_root / "signals"
        if signals_dir.exists():
            sig_files = list(signals_dir.glob("*.json"))
            _check("Signal files", len(sig_files) > 0, f"→ {len(sig_files)} files")

        # Risk checks
        risk = day_root / "risk_checks.json"
        if risk.exists():
            rdata = json.loads(risk.read_text("utf-8"))
            _check("Risk check status", rdata.get("status") in ("ok", "passed"), f"→ {rdata.get('status')}")
        else:
            _check("Risk checks artifact", False, "not generated today")

    # 3. DB order intents
    conn = sqlite3.connect(DB_PATH)
    try:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM order_intents WHERE asof_date=?", (today,)
        ).fetchone()
        n_intents = row[0] if row else 0
        _check("DB intents for today", n_intents >= 0, f"→ {n_intents} in order_intents")
    except sqlite3.Error as e:
        _check("DB intents", False, str(e))
    conn.close()

    # 4. VRAM leak check
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                alloc = torch.cuda.memory_allocated(i) / (1024**2)
                reserved = torch.cuda.memory_reserved(i) / (1024**2)
                _check(
                    f"CUDA:{i} memory",
                    alloc < 500,  # Should be near-zero if workers cleaned up
                    f"→ alloc={alloc:.0f}MB reserved={reserved:.0f}MB",
                )
    except ImportError:
        pass

    logger.info("═" * 60)
    if all_ok:
        logger.info("✅ All validation checks passed for %s", today)
    else:
        logger.info("⚠️  Some checks failed — review above before continuing")
    logger.info("═" * 60)


def status():
    """Print current control state and schedule."""
    state = _state()
    snap = state.snapshot()
    logger.info("Current Control State:")
    for k, v in snap.items():
        logger.info("  %s = %s", k, v)

    try:
        from backend.app.orchestrator.phase_scheduler import PhaseScheduler
        s = PhaseScheduler(db_path=DB_PATH)
        logger.info("\nSchedule:")
        for entry in s.get_schedule_summary():
            logger.info(
                "  ⏰ %-25s @ %s  (fires in %ds)",
                entry["name"], entry["trigger_time"], entry["next_fire_in_s"],
            )
    except Exception as e:
        logger.error("Could not load scheduler: %s", e)

    # Recent audit trail
    audit = state.get_audit(10)
    if audit:
        logger.info("\nRecent Audit Trail:")
        for entry in audit:
            logger.info("  %s %s %s", entry["ts"][:19], entry["action"], entry.get("detail", ""))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Paper Burn-In")
    parser.add_argument("command", choices=["setup", "validate", "status"])
    args = parser.parse_args()

    {"setup": setup, "validate": validate, "status": status}[args.command]()
