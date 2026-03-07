"""Intent aggregation barrier — collects and validates TargetIntents.

After all ML sleeves in the DAG finish producing ``intents.json`` files,
this module reads them, deserializes into ``TargetIntent`` objects, and
atomically passes them through the risk gateway.

This is the "aggregation barrier" between **inference** (Phases 1-2 of the
pipeline) and **execution** (routing via ``route_phase_orders``).

Shadow Mode (Phase 4)
---------------------
When ``SHADOW_SLEEVES`` is configured, intents from shadow sleeves are
intercepted and logged to a local JSON ledger instead of being routed
to the broker.  This enables statistical validation without risking
real capital.
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from backend.app.core.risk_gateway import validate_and_store_intents
from backend.app.core.schemas import TargetIntent

logger = logging.getLogger(__name__)

# ── Shadow Mode Configuration ────────────────────────────────────────────
# Sleeves listed here will have their intents intercepted and logged
# to the shadow ledger instead of being submitted to the broker.
# Set via env var: SHADOW_SLEEVES="STATARB,VRP,SELECTOR,FUTURES_OVERNIGHT"
_SHADOW_SLEEVES_RAW = os.getenv("SHADOW_SLEEVES", "")
SHADOW_SLEEVES: set[str] = {
    s.strip().upper() for s in _SHADOW_SLEEVES_RAW.split(",") if s.strip()
}
SHADOW_LEDGER_DIR = Path(
    os.getenv("SHADOW_LEDGER_DIR", "backend/artifacts/shadow_ledger")
)


def _save_to_shadow_ledger(
    intent: TargetIntent, asof_date: str
) -> None:
    """Write an intercepted intent to the shadow ledger."""
    ledger_dir = SHADOW_LEDGER_DIR / asof_date
    ledger_dir.mkdir(parents=True, exist_ok=True)
    ledger_file = ledger_dir / "shadow_intents.jsonl"

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "sleeve": getattr(intent, "sleeve", "unknown"),
        "symbol": getattr(intent, "symbol", "unknown"),
        "target_weight": getattr(intent, "target_weight", 0.0),
        "execution_phase": getattr(getattr(intent, "execution_phase", None), "value", None),
        "intent": intent.model_dump() if hasattr(intent, "model_dump") else intent.__dict__,
    }

    with open(ledger_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

    logger.info(
        "[SHADOW MODE] Intercepted intent %s from %s (w=%.4f) — logged to %s",
        record["symbol"], record["sleeve"], float(record["target_weight"]), ledger_file,
    )


def collect_and_validate_intents(
    artifact_root: Path | str,
    db_path: Path | str,
    asof_date: str,
) -> dict[str, Any]:
    """Scan ``artifact_root/intents/`` for all ``*_intents.json`` files,
    deserialize into ``TargetIntent`` objects, and push through the risk
    gateway atomically.

    Parameters
    ----------
    artifact_root : Path
        Day-level artifact directory (e.g. ``artifacts/orchestrator/2026-02-28``).
    db_path : Path
        SQLite state database path.
    asof_date : str
        Trading date (``YYYY-MM-DD``).

    Returns
    -------
    dict with ``status``, ``n_collected``, ``n_validated``, ``risk_result``.
    """
    artifact_root = Path(artifact_root)
    intents_dir = artifact_root / "intents"
    targets_dir = artifact_root / "targets"

    all_intents: list[TargetIntent] = []
    files_read: list[str] = []
    target_intent_files = 0

    # Scan for plugin-generated intent files
    for search_dir in [intents_dir, targets_dir]:
        if not search_dir.exists():
            continue
        for p in sorted(search_dir.glob("*_intents.json")):
            if search_dir == targets_dir:
                target_intent_files += 1
            try:
                raw = json.loads(p.read_text(encoding="utf-8"))
                entries = raw if isinstance(raw, list) else raw.get("intents", [])
                for entry in entries:
                    # Allow intents to omit asof_date — inject it
                    if "asof_date" not in entry:
                        entry["asof_date"] = asof_date
                    intent = TargetIntent(**entry)
                    all_intents.append(intent)
                files_read.append(str(p))
            except Exception as exc:
                logger.error("Failed to parse intent file %s: %s", p, exc)

    if not all_intents:
        logger.info("No intents collected from %s — nothing to validate", artifact_root)
        return {
            "status": "no_intents",
            "n_collected": 0,
            "n_validated": 0,
            "files_read": files_read,
            "n_target_intent_files": target_intent_files,
        }

    logger.info(
        "Collected %d intents from %d files — submitting to risk gateway",
        len(all_intents), len(files_read),
    )

    # ── Shadow Mode Interception ─────────────────────────────────────
    if SHADOW_SLEEVES:
        live_intents = []
        shadow_count = 0
        for intent in all_intents:
            sleeve = getattr(intent, "sleeve", "").upper()
            if sleeve in SHADOW_SLEEVES:
                _save_to_shadow_ledger(intent, asof_date)
                shadow_count += 1
            else:
                live_intents.append(intent)
        if shadow_count:
            logger.info(
                "[SHADOW MODE] Intercepted %d intents, routing %d live",
                shadow_count, len(live_intents),
            )
        all_intents = live_intents

    if not all_intents:
        return {
            "status": "ok",
            "n_collected": len(files_read),
            "n_validated": 0,
            "n_shadowed": shadow_count if SHADOW_SLEEVES else 0,
            "files_read": files_read,
            "n_target_intent_files": target_intent_files,
        }

    try:
        risk_result = validate_and_store_intents(db_path, all_intents)
        return {
            "status": "ok",
            "n_collected": len(all_intents),
            "n_validated": risk_result.get("n_stored", len(all_intents)),
            "files_read": files_read,
            "n_target_intent_files": target_intent_files,
            "risk_result": risk_result,
        }
    except RuntimeError as exc:
        logger.error("Risk gateway REJECTED intents: %s", exc)
        return {
            "status": "rejected",
            "n_collected": len(all_intents),
            "n_validated": 0,
            "files_read": files_read,
            "n_target_intent_files": target_intent_files,
            "error": str(exc),
        }
