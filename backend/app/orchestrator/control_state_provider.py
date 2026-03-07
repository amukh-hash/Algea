"""Unified control-state provider with in-memory cache + DB write-through.

PR-5 scope: default runtime path only.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ControlStateProvider:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._cache = {
            "schema_version": "control_state.v1",
            "snapshot_id": str(uuid.uuid4()),
            "paused": False,
            "vol_regime_override": None,
            "blocked_symbols": [],
            "frozen_sleeves": [],
            "gross_exposure_cap": 1.5,
            "execution_mode": "paper",
        }
        self._audit_log: list[dict[str, Any]] = []
        self._tick_snapshot_ids: dict[str, str] = {}
        self._ensure_storage()
        self._load_from_db_into_cache()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_storage(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_control_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    is_paused BOOLEAN NOT NULL DEFAULT 0,
                    vol_regime VARCHAR(50) DEFAULT 'NORMAL',
                    gross_exposure_cap REAL DEFAULT 1.5,
                    execution_mode VARCHAR(20) NOT NULL DEFAULT 'paper',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute("INSERT OR IGNORE INTO app_control_state (id) VALUES (1)")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS app_control_state_ext (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    snapshot_id TEXT NOT NULL,
                    blocked_symbols_json TEXT NOT NULL DEFAULT '[]',
                    frozen_sleeves_json TEXT NOT NULL DEFAULT '[]',
                    audit_log_json TEXT NOT NULL DEFAULT '[]',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO app_control_state_ext (id, snapshot_id) VALUES (1, ?)",
                (self._cache["snapshot_id"],),
            )

    def _load_from_db_into_cache(self) -> None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT is_paused, vol_regime, gross_exposure_cap, execution_mode FROM app_control_state WHERE id=1"
            ).fetchone()
            ext = conn.execute(
                "SELECT snapshot_id, blocked_symbols_json, frozen_sleeves_json, audit_log_json FROM app_control_state_ext WHERE id=1"
            ).fetchone()

        if row:
            self._cache["paused"] = bool(row["is_paused"])
            regime = row["vol_regime"]
            self._cache["vol_regime_override"] = None if regime in (None, "NORMAL") else str(regime)
            self._cache["gross_exposure_cap"] = float(row["gross_exposure_cap"]) if row["gross_exposure_cap"] is not None else None
            self._cache["execution_mode"] = str(row["execution_mode"])

        if ext:
            self._cache["snapshot_id"] = str(ext["snapshot_id"] or self._cache["snapshot_id"])
            self._cache["blocked_symbols"] = sorted({str(s).upper() for s in json.loads(ext["blocked_symbols_json"] or "[]")})
            self._cache["frozen_sleeves"] = sorted({str(s) for s in json.loads(ext["frozen_sleeves_json"] or "[]")})
            self._audit_log = list(json.loads(ext["audit_log_json"] or "[]"))

    def _write_through(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE app_control_state
                SET is_paused=?, vol_regime=?, gross_exposure_cap=?, execution_mode=?, updated_at=?
                WHERE id=1
                """,
                (
                    int(bool(self._cache["paused"])),
                    self._cache["vol_regime_override"] or "NORMAL",
                    self._cache["gross_exposure_cap"],
                    self._cache["execution_mode"],
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
            conn.execute(
                """
                UPDATE app_control_state_ext
                SET snapshot_id=?, blocked_symbols_json=?, frozen_sleeves_json=?, audit_log_json=?, updated_at=?
                WHERE id=1
                """,
                (
                    self._cache["snapshot_id"],
                    json.dumps(list(self._cache["blocked_symbols"])),
                    json.dumps(list(self._cache["frozen_sleeves"])),
                    json.dumps(self._audit_log[-500:]),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def _audit(self, action: str, detail: dict[str, Any] | None = None) -> None:
        self._audit_log.append(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "action": action,
                "detail": detail or {},
            }
        )
        if len(self._audit_log) > 500:
            self._audit_log = self._audit_log[-500:]

    def snapshot(self, *, consumer: str = "unknown", tick_id: str | None = None) -> dict[str, Any]:
        with self._lock:
            snap = dict(self._cache)
            snap["read_consumer"] = consumer
            snap["read_tick_id"] = tick_id
            if tick_id:
                prior = self._tick_snapshot_ids.get(tick_id)
                if prior and prior != str(snap.get("snapshot_id")):
                    logger.warning(
                        "control_state_snapshot_mismatch tick_id=%s prior=%s now=%s consumer=%s",
                        tick_id,
                        prior,
                        snap.get("snapshot_id"),
                        consumer,
                    )
                self._tick_snapshot_ids[tick_id] = str(snap.get("snapshot_id"))
            logger.info(
                "control_state_read consumer=%s snapshot_id=%s paused=%s execution_mode=%s",
                consumer,
                snap.get("snapshot_id"),
                snap.get("paused"),
                snap.get("execution_mode"),
            )
            return snap

    def mutate(self, action: str, updater) -> dict[str, Any]:
        with self._lock:
            prev_cache = deepcopy(self._cache)
            prev_audit = deepcopy(self._audit_log)
            updater(self._cache)
            self._cache["snapshot_id"] = str(uuid.uuid4())
            self._audit(action)
            try:
                self._write_through()
                logger.info(
                    "control_state_write_through_ok action=%s snapshot_id=%s",
                    action,
                    self._cache["snapshot_id"],
                )
            except Exception as exc:
                self._cache = prev_cache
                self._audit_log = prev_audit
                logger.error(
                    "control_state_write_through_failed action=%s error=%s",
                    action,
                    exc,
                    exc_info=True,
                )
                raise RuntimeError(f"control state write-through failed for action={action}: {exc}") from exc
            return dict(self._cache)

    def set_paused(self, paused: bool) -> dict[str, Any]:
        return self.mutate("pause" if paused else "resume", lambda s: s.__setitem__("paused", bool(paused)))

    def set_vol_regime(self, regime: str | None) -> dict[str, Any]:
        return self.mutate("vol_regime_override", lambda s: s.__setitem__("vol_regime_override", regime))

    def set_blocked_symbols(self, symbols: list[str]) -> dict[str, Any]:
        return self.mutate("blocked_symbols", lambda s: s.__setitem__("blocked_symbols", sorted({str(x).upper() for x in symbols})))

    def set_frozen_sleeves(self, sleeves: list[str]) -> dict[str, Any]:
        return self.mutate("frozen_sleeves", lambda s: s.__setitem__("frozen_sleeves", sorted({str(x) for x in sleeves})))

    def set_exposure_cap(self, cap: float | None) -> dict[str, Any]:
        return self.mutate("exposure_cap", lambda s: s.__setitem__("gross_exposure_cap", None if cap is None else float(cap)))

    def set_execution_mode(self, mode: str) -> dict[str, Any]:
        if mode not in ("noop", "paper", "ibkr"):
            raise ValueError(f"Invalid mode: {mode}")
        return self.mutate("execution_mode", lambda s: s.__setitem__("execution_mode", mode))

    def audit(self, action: str, detail: dict[str, Any] | None = None) -> None:
        with self._lock:
            prev_audit = deepcopy(self._audit_log)
            self._audit(action, detail)
            try:
                self._write_through()
            except Exception as exc:
                self._audit_log = prev_audit
                raise RuntimeError(f"control state audit write-through failed: {exc}") from exc

    def get_audit(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock:
            return list(reversed(self._audit_log[-limit:]))


_PROVIDER_REGISTRY: dict[str, ControlStateProvider] = {}
_PROVIDER_LOCK = threading.Lock()


def get_control_state_provider(db_path: Path) -> ControlStateProvider:
    key = str(Path(db_path).resolve())
    with _PROVIDER_LOCK:
        if key not in _PROVIDER_REGISTRY:
            _PROVIDER_REGISTRY[key] = ControlStateProvider(Path(db_path))
        return _PROVIDER_REGISTRY[key]
