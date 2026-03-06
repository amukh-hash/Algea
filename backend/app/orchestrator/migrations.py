from __future__ import annotations

import sqlite3

CURRENT_SCHEMA_VERSION = 3


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone()
    return row is not None


def _column_exists(conn: sqlite3.Connection, table_name: str, column_name: str) -> bool:
    cols = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return any(c[1] == column_name for c in cols)


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


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Add last_success_at column for per-job cooldown tracking."""
    if not _column_exists(conn, "jobs", "last_success_at"):
        conn.execute("ALTER TABLE jobs ADD COLUMN last_success_at TEXT")
    conn.execute("UPDATE schema_version SET version=2")


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """Durable control state + order intents for phase-aware risk gateway.

    Resolves F1 (core risk bypass), F2 (volatile control state),
    and F4 (idempotent intent UPSERT).
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS app_control_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            is_paused BOOLEAN NOT NULL DEFAULT 0,
            vol_regime VARCHAR(50) DEFAULT 'NORMAL',
            gross_exposure_cap REAL DEFAULT 1.5,
            execution_mode VARCHAR(20) NOT NULL DEFAULT 'paper',
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("INSERT OR IGNORE INTO app_control_state (id) VALUES (1)")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS order_intents (
            asof_date VARCHAR(10) NOT NULL,
            execution_phase VARCHAR(20) NOT NULL,
            sleeve VARCHAR(50) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            target_weight REAL NOT NULL,
            asset_class VARCHAR(20) NOT NULL,
            multiplier REAL NOT NULL DEFAULT 1.0,
            status VARCHAR(20) NOT NULL DEFAULT 'PENDING',
            PRIMARY KEY (asof_date, execution_phase, sleeve, symbol)
        )
    """)
    conn.execute("UPDATE schema_version SET version=3")


def apply_migrations(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    if not _table_exists(conn, "schema_version"):
        _create_v1(conn)
        _migrate_v1_to_v2(conn)
        _migrate_v2_to_v3(conn)
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

    # explicit incremental migration chain
    while version < CURRENT_SCHEMA_VERSION:
        if version == 0:
            _create_v1(conn)
            version = 1
            continue
        if version == 1:
            _migrate_v1_to_v2(conn)
            version = 2
            continue
        if version == 2:
            _migrate_v2_to_v3(conn)
            version = 3
            continue
        raise RuntimeError(f"No migration path from schema version {version}")

    # ensure v1 tables/indexes exist deterministically
    _create_v1(conn)
    if not _column_exists(conn, "jobs", "last_success_at"):
        conn.execute("ALTER TABLE jobs ADD COLUMN last_success_at TEXT")
    conn.execute("UPDATE schema_version SET version=?", (CURRENT_SCHEMA_VERSION,))
    conn.commit()
