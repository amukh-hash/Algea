from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

from .migrations import apply_migrations


def _utc_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


class StateStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db(db_path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self, db_path: Path) -> None:
        with sqlite3.connect(db_path) as conn:
            apply_migrations(conn)

    def get_job_status(self, asof_date: str, session: str, job_name: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT status FROM jobs WHERE asof_date=? AND session=? AND job_name=?",
                (asof_date, session, job_name),
            ).fetchone()
        return None if row is None else str(row["status"])

    def _upsert_job(self, **fields: Any) -> None:
        keys = [
            "run_id",
            "asof_date",
            "session",
            "job_name",
            "status",
            "started_at",
            "ended_at",
            "exit_code",
            "error_summary",
            "stdout_path",
            "stderr_path",
            "artifacts_json",
        ]
        values = [fields.get(k) for k in keys]
        with self._connect() as conn:
            conn.execute("BEGIN")
            conn.execute(
                """
                INSERT INTO jobs(run_id, asof_date, session, job_name, status, started_at, ended_at, exit_code, error_summary, stdout_path, stderr_path, artifacts_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(asof_date, session, job_name) DO UPDATE SET
                    run_id=excluded.run_id,
                    status=excluded.status,
                    started_at=COALESCE(excluded.started_at, jobs.started_at),
                    ended_at=excluded.ended_at,
                    exit_code=excluded.exit_code,
                    error_summary=excluded.error_summary,
                    stdout_path=excluded.stdout_path,
                    stderr_path=excluded.stderr_path,
                    artifacts_json=excluded.artifacts_json
                """,
                values,
            )
            conn.execute("COMMIT")

    def mark_job_running(self, run_id: str, asof_date: str, session: str, job_name: str) -> None:
        self._upsert_job(
            run_id=run_id,
            asof_date=asof_date,
            session=session,
            job_name=job_name,
            status="running",
            started_at=_utc_iso(),
            artifacts_json=json.dumps([]),
        )

    def mark_job_success(
        self,
        run_id: str,
        asof_date: str,
        session: str,
        job_name: str,
        stdout_path: str,
        stderr_path: str,
        artifacts: list[str],
    ) -> None:
        self._upsert_job(
            run_id=run_id,
            asof_date=asof_date,
            session=session,
            job_name=job_name,
            status="success",
            ended_at=_utc_iso(),
            exit_code=0,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            artifacts_json=json.dumps(artifacts),
        )

    def mark_job_failed(
        self,
        run_id: str,
        asof_date: str,
        session: str,
        job_name: str,
        error_summary: str,
        exit_code: int,
        stdout_path: str,
        stderr_path: str,
    ) -> None:
        self._upsert_job(
            run_id=run_id,
            asof_date=asof_date,
            session=session,
            job_name=job_name,
            status="failed",
            ended_at=_utc_iso(),
            exit_code=exit_code,
            error_summary=error_summary,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            artifacts_json=json.dumps([]),
        )

    def mark_job_skipped(self, run_id: str, asof_date: str, session: str, job_name: str, reason: str) -> None:
        self._upsert_job(
            run_id=run_id,
            asof_date=asof_date,
            session=session,
            job_name=job_name,
            status="skipped",
            ended_at=_utc_iso(),
            error_summary=reason,
            artifacts_json=json.dumps([]),
        )

    def create_run_record(self, run_id: str, asof_date: str, session: str, meta: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO runs(run_id, asof_date, session, started_at, status, meta_json) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, asof_date, session, _utc_iso(), "running", json.dumps(meta)),
            )

    def update_run_record(self, run_id: str, status: str, meta: dict[str, Any]) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE runs SET ended_at=?, status=?, meta_json=? WHERE run_id=?",
                (_utc_iso(), status, json.dumps(meta), run_id),
            )

    def list_runs(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]
