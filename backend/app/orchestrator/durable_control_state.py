"""Durable control state backed by SQLite.

Replaces the memory-only ``control_state.py`` singleton.  All overrides
(pause, vol regime, exposure cap, execution mode) are persisted to
``app_control_state`` in ``state.sqlite3`` and survive process restarts.

Resolves **F2** (volatile control state).
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DurableControlState:
    """Thread-safe, crash-durable control state manager.

    Every mutation writes through to SQLite immediately.  Reads use
    a short-lived connection per call to avoid stale caches.

    Parameters
    ----------
    db_path : Path
        Path to the orchestrator SQLite database.
    """

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._lock = threading.Lock()
        self._audit_log: list[dict[str, Any]] = []

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ── Reads ────────────────────────────────────────────────────────

    def snapshot(self) -> dict[str, Any]:
        """Return current control state as a dict (thread-safe)."""
        with self._lock:
            conn = self._connect()
            try:
                row = conn.execute(
                    "SELECT is_paused, vol_regime, gross_exposure_cap, execution_mode "
                    "FROM app_control_state WHERE id=1"
                ).fetchone()
                if row is None:
                    return {
                        "paused": False,
                        "vol_regime": "NORMAL",
                        "gross_exposure_cap": 1.5,
                        "execution_mode": "paper",
                    }
                return {
                    "paused": bool(row["is_paused"]),
                    "vol_regime": row["vol_regime"],
                    "gross_exposure_cap": float(row["gross_exposure_cap"]),
                    "execution_mode": row["execution_mode"],
                }
            finally:
                conn.close()

    def is_paused(self) -> bool:
        return self.snapshot().get("paused", False)

    def get_exposure_cap(self) -> float:
        return self.snapshot().get("gross_exposure_cap", 1.5)

    def get_execution_mode(self) -> str:
        return self.snapshot().get("execution_mode", "paper")

    # ── Mutations ────────────────────────────────────────────────────

    def _update(self, column: str, value: Any, action: str) -> None:
        with self._lock:
            conn = self._connect()
            try:
                conn.execute(
                    f"UPDATE app_control_state SET {column}=?, updated_at=? WHERE id=1",
                    (value, datetime.now(timezone.utc).isoformat()),
                )
                conn.commit()
                self._audit(action, {column: value})
            finally:
                conn.close()

    def set_paused(self, paused: bool) -> None:
        self._update("is_paused", int(paused), "pause" if paused else "resume")

    def set_vol_regime(self, regime: str) -> None:
        self._update("vol_regime", regime, "vol_regime_override")

    def set_exposure_cap(self, cap: float) -> None:
        self._update("gross_exposure_cap", cap, "exposure_cap")

    def set_execution_mode(self, mode: str) -> None:
        if mode not in ("noop", "paper", "ibkr"):
            raise ValueError(f"Invalid execution mode: {mode}")
        self._update("execution_mode", mode, "execution_mode")

    # ── Audit trail (in-memory, matches legacy interface) ────────────

    def _audit(self, action: str, detail: dict[str, Any]) -> None:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action": action,
            "detail": detail,
        }
        self._audit_log.append(entry)
        if len(self._audit_log) > 500:
            self._audit_log = self._audit_log[-500:]

    def get_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self._audit_log[-limit:]))
