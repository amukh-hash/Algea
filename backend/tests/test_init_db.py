"""Tests for idempotent database initialization.

Verifies that init_db:
  - Creates the database with WAL journal mode
  - Creates ece_tracking, runs, and jobs tables
  - Is idempotent (safe to call multiple times)
  - Sets busy_timeout and foreign_keys
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


class TestInitDb:

    def test_creates_tables(self, tmp_path):
        """init_db should create ece_tracking, runs, and jobs tables."""
        from backend.app.db.init_db import init_db

        db_path = tmp_path / "test.sqlite3"
        conn = init_db(db_path)

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = {row["name"] for row in tables}

        assert "ece_tracking" in table_names
        assert "runs" in table_names
        assert "jobs" in table_names
        conn.close()

    def test_wal_mode(self, tmp_path):
        """Database should use WAL journal mode."""
        from backend.app.db.init_db import init_db

        db_path = tmp_path / "test.sqlite3"
        conn = init_db(db_path)

        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        conn.close()

    def test_busy_timeout(self, tmp_path):
        """busy_timeout should be 5000ms."""
        from backend.app.db.init_db import init_db

        db_path = tmp_path / "test.sqlite3"
        conn = init_db(db_path)

        timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]
        assert timeout == 5000
        conn.close()

    def test_foreign_keys_on(self, tmp_path):
        """Foreign keys should be enforced."""
        from backend.app.db.init_db import init_db

        db_path = tmp_path / "test.sqlite3"
        conn = init_db(db_path)

        fk = conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1
        conn.close()

    def test_idempotent(self, tmp_path):
        """Calling init_db twice should not raise or duplicate tables."""
        from backend.app.db.init_db import init_db

        db_path = tmp_path / "test.sqlite3"
        conn1 = init_db(db_path)
        conn1.close()

        # Second call should succeed without error
        conn2 = init_db(db_path)

        tables = conn2.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [row["name"] for row in tables]

        # Each table should appear exactly once
        assert table_names.count("ece_tracking") == 1
        assert table_names.count("runs") == 1
        assert table_names.count("jobs") == 1
        conn2.close()

    def test_ece_tracking_schema(self, tmp_path):
        """ece_tracking should have the correct columns."""
        from backend.app.db.init_db import init_db

        db_path = tmp_path / "test.sqlite3"
        conn = init_db(db_path)

        info = conn.execute("PRAGMA table_info(ece_tracking)").fetchall()
        col_names = {row["name"] for row in info}

        expected = {"trade_id", "sleeve", "confidence_bin", "predicted_probability",
                    "actual_outcome", "recorded_at"}
        assert expected == col_names
        conn.close()

    def test_creates_parent_dirs(self, tmp_path):
        """init_db should create parent directories if missing."""
        from backend.app.db.init_db import init_db

        db_path = tmp_path / "deep" / "nested" / "dir" / "test.sqlite3"
        conn = init_db(db_path)
        assert db_path.exists()
        conn.close()

    def test_indices_created(self, tmp_path):
        """Indices should be created on ece_tracking and jobs."""
        from backend.app.db.init_db import init_db

        db_path = tmp_path / "test.sqlite3"
        conn = init_db(db_path)

        indices = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
        idx_names = {row["name"] for row in indices}

        assert "idx_ece_sleeve_bin" in idx_names
        assert "idx_jobs_run_id" in idx_names
        assert "idx_jobs_name" in idx_names
        conn.close()
