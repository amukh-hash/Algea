from __future__ import annotations

import sqlite3

CURRENT_SCHEMA_VERSION = 1


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()
    return row is not None


def _create_v1(conn: sqlite3.Connection) -> None:
    conn.execute("CREATE TABLE IF NOT EXISTS schema_version(version INTEGER NOT NULL)")
    conn.execute("DELETE FROM schema_version")
    conn.execute("INSERT INTO schema_version(version) VALUES (1)")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS runs(
            run_id TEXT PRIMARY KEY,
            asof_date TEXT,
            session TEXT,
            started_at TEXT,
            ended_at TEXT,
            status TEXT,
            meta_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS jobs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT,
            asof_date TEXT,
            session TEXT,
            job_name TEXT,
            status TEXT,
            started_at TEXT,
            ended_at TEXT,
            exit_code INTEGER,
            error_summary TEXT,
            stdout_path TEXT,
            stderr_path TEXT,
            artifacts_json TEXT,
            UNIQUE(asof_date, session, job_name)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_asof_session ON jobs(asof_date, session)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_name ON jobs(job_name)")


def apply_migrations(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    if not _table_exists(conn, "schema_version"):
        _create_v1(conn)
        conn.commit()
        return

    row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    if row is None:
        conn.execute("DELETE FROM schema_version")
        conn.execute("INSERT INTO schema_version(version) VALUES (1)")
        row = (1,)

    version = int(row[0])
    if version > CURRENT_SCHEMA_VERSION:
        raise RuntimeError(
            f"Schema version {version} is newer than supported {CURRENT_SCHEMA_VERSION}. "
            "Upgrade orchestrator code before continuing."
        )

    # explicit incremental migration chain placeholder
    while version < CURRENT_SCHEMA_VERSION:
        if version == 0:
            _create_v1(conn)
            version = 1
            conn.execute("UPDATE schema_version SET version=1")
            continue
        raise RuntimeError(f"No migration path from schema version {version}")

    # ensure v1 tables/indexes exist deterministically
    _create_v1(conn)
    conn.execute("UPDATE schema_version SET version=?", (CURRENT_SCHEMA_VERSION,))
    conn.commit()
