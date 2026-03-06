"""Test: ECE tracking schema has all columns required by 08_t1_resolution.py.

Validates that the DDL in 00_init_state.py creates the ece_tracking table
with all 14 columns, including the 8 T+1 resolution columns added for
corporate action adjustment.
"""
from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path

import pytest


_DDL = textwrap.dedent("""\
    CREATE TABLE IF NOT EXISTS ece_tracking (
        trade_id              TEXT PRIMARY KEY,
        sleeve                VARCHAR(50) NOT NULL,
        confidence_bin        VARCHAR(20) NOT NULL,
        predicted_probability REAL NOT NULL,
        actual_outcome        INTEGER,
        recorded_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        prediction_date       TEXT,
        predicted_direction   REAL,
        t0_price              REAL,
        symbol                TEXT,
        t1_price              REAL,
        resolved_at           TIMESTAMP,
        cum_split_factor      REAL DEFAULT 1.0,
        cum_dividend_factor   REAL DEFAULT 1.0
    );
""")


# Required columns for 08_t1_resolution.py queries and writes
_REQUIRED_COLUMNS = {
    "trade_id", "sleeve", "confidence_bin", "predicted_probability",
    "actual_outcome", "recorded_at",
    # T+1 resolution columns
    "prediction_date", "predicted_direction", "t0_price", "symbol",
    "t1_price", "resolved_at", "cum_split_factor", "cum_dividend_factor",
}


def _create_schema(db_path: Path) -> None:
    conn = sqlite3.connect(db_path)
    conn.executescript(_DDL)
    conn.close()


def test_ece_tracking_schema_has_all_columns(tmp_path: Path) -> None:
    """Verify ECE table has every column that 08_t1_resolution.py queries."""
    db = tmp_path / "test_state.sqlite3"
    _create_schema(db)

    conn = sqlite3.connect(db)
    rows = conn.execute("PRAGMA table_info(ece_tracking)").fetchall()
    conn.close()

    actual_columns = {row[1] for row in rows}
    missing = _REQUIRED_COLUMNS - actual_columns
    assert not missing, (
        f"ece_tracking table is missing columns required by 08_t1_resolution.py: {missing}"
    )


def test_ece_tracking_defaults(tmp_path: Path) -> None:
    """Verify corporate action columns default to 1.0 (no adjustment)."""
    db = tmp_path / "test_state.sqlite3"
    _create_schema(db)

    conn = sqlite3.connect(db)
    conn.execute(
        "INSERT INTO ece_tracking (trade_id, sleeve, confidence_bin, predicted_probability) "
        "VALUES ('test_1', 'kronos', '0.80-0.90', 0.85)"
    )
    conn.commit()

    row = conn.execute(
        "SELECT cum_split_factor, cum_dividend_factor FROM ece_tracking WHERE trade_id = 'test_1'"
    ).fetchone()
    conn.close()

    assert row[0] == 1.0, f"cum_split_factor default should be 1.0, got {row[0]}"
    assert row[1] == 1.0, f"cum_dividend_factor default should be 1.0, got {row[1]}"
