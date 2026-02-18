from __future__ import annotations

import json
import os
import socket
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
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
            "CREATE TABLE IF NOT EXISTS locks(name TEXT PRIMARY KEY, owner_json TEXT NOT NULL, acquired_at TEXT NOT NULL)"
        )
        return conn

    @contextmanager
    def acquire(self, name: str):
        owner = LockOwner(pid=os.getpid(), hostname=socket.gethostname(), started_at=datetime.now(timezone.utc).isoformat())
        conn = self._connect()
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT owner_json FROM locks WHERE name=?", (name,)).fetchone()
            if row is not None:
                existing = json.loads(row["owner_json"])
                same_host = existing.get("hostname") == socket.gethostname()
                existing_pid = int(existing.get("pid", -1))
                if same_host and not _pid_alive(existing_pid):
                    conn.execute("DELETE FROM locks WHERE name=?", (name,))
                else:
                    conn.execute("ROLLBACK")
                    raise RuntimeError(f"Lock '{name}' is held by {existing}")
            conn.execute(
                "INSERT INTO locks(name, owner_json, acquired_at) VALUES (?, ?, ?)",
                (name, owner.to_json(), datetime.now(timezone.utc).isoformat()),
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
