"""Intraday portfolio value history — snapshots every 60s.

Stores snapshots in an in-memory ring buffer and persists to disk.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter

logger = logging.getLogger("algae.api.portfolio_history")

router = APIRouter(prefix="/api/control", tags=["control"])

_ARTIFACT_ROOT = Path("backend/artifacts/orchestrator")
_MAX_POINTS = 480  # 8 hours at 1-per-minute
_SNAPSHOT_INTERVAL = 60  # seconds

# In-memory ring buffer: list of { ts, total_value, core, vrp, selector, cash }
_history: deque[dict[str, Any]] = deque(maxlen=_MAX_POINTS)
_snapshot_thread: threading.Thread | None = None
_running = False


def _today() -> str:
    return date.today().isoformat()


def _history_path() -> Path:
    p = _ARTIFACT_ROOT / _today() / "portfolio_history.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def _load_history() -> None:
    """Load persisted history on startup."""
    path = _history_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _history.clear()
            for pt in data:
                _history.append(pt)
            logger.info("Loaded %d history points from %s", len(_history), path)
        except Exception as exc:
            logger.warning("Failed to load history: %s", exc)


def _persist_history() -> None:
    """Save history to disk."""
    try:
        path = _history_path()
        path.write_text(json.dumps(list(_history), indent=1), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to persist history: %s", exc)


def _take_snapshot() -> None:
    """Take a single portfolio value snapshot."""
    try:
        from backend.app.api.control_routes import get_portfolio_summary
        summary = get_portfolio_summary()

        # Aggregate by sleeve
        sleeve_values: dict[str, float] = {"core": 0, "vrp": 0, "selector": 0}
        for h in summary.get("holdings", []):
            sleeve = h.get("sleeve", "selector")
            sleeve_values[sleeve] = sleeve_values.get(sleeve, 0) + h.get("margin_posted", 0) + h.get("unrealized_pnl", 0)

        point = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "total_value": summary.get("total_value", 0),
            "cash": summary.get("cash", 0),
            "core": round(sleeve_values.get("core", 0), 2),
            "vrp": round(sleeve_values.get("vrp", 0), 2),
            "selector": round(sleeve_values.get("selector", 0), 2),
            "total_pnl": summary.get("total_pnl", 0),
            "unrealized_pnl": summary.get("total_unrealized_pnl", 0),
        }
        _history.append(point)
        _persist_history()
    except Exception as exc:
        logger.warning("Snapshot failed: %s", exc)


def _snapshot_loop() -> None:
    """Background thread that takes snapshots every 60s."""
    global _running
    _load_history()
    while _running:
        _take_snapshot()
        time.sleep(_SNAPSHOT_INTERVAL)


def start_snapshots() -> None:
    """Start the background snapshot thread."""
    global _snapshot_thread, _running
    if _snapshot_thread and _snapshot_thread.is_alive():
        return
    _running = True
    _snapshot_thread = threading.Thread(target=_snapshot_loop, daemon=True, name="portfolio-history")
    _snapshot_thread.start()
    logger.info("Portfolio history snapshots started (every %ds)", _SNAPSHOT_INTERVAL)


def stop_snapshots() -> None:
    """Stop the background snapshot thread."""
    global _running
    _running = False


@router.get("/portfolio-history")
def get_portfolio_history() -> dict[str, Any]:
    """Return intraday portfolio value history."""
    # Ensure snapshots are running
    start_snapshots()
    return {
        "asof_date": _today(),
        "points": list(_history),
        "count": len(_history),
    }
