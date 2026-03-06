"""Idempotent SQLite database initialization for the Algaie backend.

Configures the database for high-concurrency asynchronous environments:
  - WAL journal mode for concurrent readers/writers
  - NORMAL synchronous for durability without fsync-per-write
  - 5-second busy timeout to prevent SQLITE_BUSY during cron overlap
  - Foreign keys enforced

Creates the canonical ``ece_tracking`` table used by the T+1 resolver
and LiveGuard Risk Engine for calibration monitoring.
"""
from __future__ import annotations

import logging
import sqlite3
from pathlib import Path

logger = logging.getLogger(__name__)

# Default database path (orchestrator state DB)
DEFAULT_DB_PATH = Path("backend/artifacts/orchestrator_state/state.sqlite3")


def init_db(db_path: Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Initialize the database with WAL mode and canonical schema.

    This function is **idempotent**: safe to call on every server startup.
    ``CREATE TABLE IF NOT EXISTS`` guards prevent duplicate-table errors,
    and ``CREATE INDEX IF NOT EXISTS`` prevents index rebuild overhead.

    Parameters
    ----------
    db_path : Path
        Path to the SQLite database file.  Parent directories are
        created automatically if missing.

    Returns
    -------
    sqlite3.Connection
        A configured connection to the initialized database.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), timeout=10)
    conn.row_factory = sqlite3.Row

    # ── PRAGMA configuration ───────────────────────────────────────
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")

    # ── ECE Tracking ───────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ece_tracking (
            trade_id TEXT PRIMARY KEY,
            sleeve VARCHAR(50) NOT NULL,
            confidence_bin VARCHAR(20) NOT NULL,
            predicted_probability REAL NOT NULL,
            actual_outcome INTEGER DEFAULT NULL,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_ece_sleeve_bin
        ON ece_tracking(sleeve, confidence_bin);
    """)

    # ── DAG Runs ───────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT PRIMARY KEY,
            asof_date TEXT NOT NULL,
            session TEXT,
            state TEXT NOT NULL DEFAULT 'PENDING',
            started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            ended_at TIMESTAMP,
            reason TEXT,
            meta_json TEXT
        );
    """)

    # ── DAG Jobs ───────────────────────────────────────────────────
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            asof_date TEXT NOT NULL,
            session TEXT,
            job_name TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            started_at TIMESTAMP,
            ended_at TIMESTAMP,
            last_success_at TIMESTAMP,
            error_summary TEXT,
            FOREIGN KEY (run_id) REFERENCES runs(run_id)
        );
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_run_id
        ON jobs(run_id);
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_jobs_name
        ON jobs(job_name);
    """)

    conn.commit()
    logger.info("Database initialized: %s (WAL mode, busy_timeout=5000)", db_path)
    return conn
