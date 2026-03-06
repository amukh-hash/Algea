from __future__ import annotations

import json
import os
import socket
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

# F7: Default TTL for distributed locks (seconds)
LOCK_TTL_S = 120


def _pid_alive(pid: int) -> bool:
    """Best-effort local PID check (fallback only — prefer TTL)."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, SystemError):
        return False


@dataclass
class LockOwner:
    pid: int
    hostname: str
    started_at: str

    def to_json(self) -> str:
        return json.dumps({"pid": self.pid, "hostname": self.hostname, "started_at": self.started_at})


class LockManager:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            "CREATE TABLE IF NOT EXISTS locks("
            "name TEXT PRIMARY KEY, owner_json TEXT NOT NULL, "
            "acquired_at TEXT NOT NULL, expires_at REAL)"
        )
        # F7: Auto-cleanup expired locks (TTL-based, cross-host safe)
        conn.execute("DELETE FROM locks WHERE expires_at IS NOT NULL AND expires_at < ?", (time.time(),))
        return conn

    @contextmanager
    def acquire(self, name: str, ttl_s: float = LOCK_TTL_S):
        owner = LockOwner(pid=os.getpid(), hostname=socket.gethostname(), started_at=datetime.now(timezone.utc).isoformat())
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            # F7: TTL-based stale lock eviction (replaces fragile os.kill PID check)
            conn.execute(
                "DELETE FROM locks WHERE name=? AND expires_at IS NOT NULL AND expires_at < ?",
                (name, time.time()),
            )
            row = conn.execute("SELECT owner_json FROM locks WHERE name=?", (name,)).fetchone()
            if row is not None:
                existing = json.loads(row["owner_json"])
                # Fallback: local PID check (same-host only)
                same_host = existing.get("hostname") == socket.gethostname()
                existing_pid = int(existing.get("pid", -1))
                if same_host and not _pid_alive(existing_pid):
                    conn.execute("DELETE FROM locks WHERE name=?", (name,))
                else:
                    conn.execute("ROLLBACK")
                    raise RuntimeError(f"Lock '{name}' is held by {existing}")
            conn.execute(
                "INSERT INTO locks(name, owner_json, acquired_at, expires_at) VALUES (?, ?, ?, ?)",
                (name, owner.to_json(), datetime.now(timezone.utc).isoformat(), time.time() + ttl_s),
            )
            conn.execute("COMMIT")
            yield owner
        finally:
            try:
                conn.execute("BEGIN IMMEDIATE")
                conn.execute("DELETE FROM locks WHERE name=?", (name,))
                conn.execute("COMMIT")
            except Exception:
                pass
            conn.close()

    def acquire_global_lock(self):
        return self.acquire("orchestrator_global")

    def acquire_job_lock(self, job_name: str):
        return self.acquire(f"job::{job_name}")
