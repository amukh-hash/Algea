"""Test ECE circuit breaker and calibration tracking.

Validates:
  1. ECE computation matches expected formula
  2. HALTED_ECE_BREACH triggers at threshold
  3. T+1 outcome resolution works correctly
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
import uuid
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.app.orchestrator.ece_tracker import (
    check_ece,
    compute_ece,
    record_prediction,
    resolve_outcome,
)


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary SQLite database with ECE schema."""
    db_path = tmp_path / "test_state.sqlite3"
    conn = sqlite3.connect(db_path)
    conn.executescript("""
        CREATE TABLE ece_tracking (
            trade_id TEXT PRIMARY KEY,
            sleeve VARCHAR(50) NOT NULL,
            confidence_bin VARCHAR(20) NOT NULL,
            predicted_probability REAL NOT NULL,
            actual_outcome INTEGER,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX idx_ece_sleeve_bin ON ece_tracking(sleeve, confidence_bin);
    """)
    conn.close()
    return db_path


class TestECETracker:
    def test_record_and_resolve(self, temp_db):
        """Record a prediction and resolve its outcome."""
        tid = record_prediction("kronos", 0.85, "0.80-0.90", db_path=temp_db)
        assert tid is not None

        resolve_outcome(tid, 1, db_path=temp_db)

        conn = sqlite3.connect(temp_db)
        row = conn.execute("SELECT actual_outcome FROM ece_tracking WHERE trade_id=?", (tid,)).fetchone()
        conn.close()
        assert row[0] == 1

    def test_ece_formula(self, temp_db):
        """ECE matches: Σ (N_bin/N_total) × |acc - conf|."""
        import random
        random.seed(42)

        # Create perfectly calibrated predictions
        for _ in range(100):
            prob = random.uniform(0.80, 0.90)
            tid = record_prediction("kronos", prob, "0.80-0.90", db_path=temp_db)
            # ~85% accuracy for 0.80-0.90 bin → well calibrated
            resolve_outcome(tid, 1 if random.random() < 0.85 else 0, db_path=temp_db)

        result = compute_ece(sleeve="kronos", min_bin_samples=10, db_path=temp_db)
        # Well-calibrated predictions should have low ECE
        assert result["ece_score"] < 0.15, f"ECE too high for calibrated predictions: {result['ece_score']}"

    def test_ece_breach_detection(self, temp_db):
        """HALTED_ECE_BREACH triggers when ECE > 0.10 for high-confidence bins."""
        # Create poorly calibrated predictions: stated 85% but only 50% accurate
        for i in range(60):
            tid = record_prediction("mera", 0.85, "0.80-0.90", db_path=temp_db)
            resolve_outcome(tid, 1 if i < 30 else 0, db_path=temp_db)  # 50% accuracy

        result = check_ece(sleeve="mera", threshold=0.10, min_samples=10, db_path=temp_db)

        # |50% - 85%| = 35% → ECE should be much > 0.10
        assert result["high_confidence_ece"] > 0.20
        assert result["is_breached"] is True
        assert result["trigger_state"] == "HALTED_ECE_BREACH"

    def test_insufficient_data_no_breach(self, temp_db):
        """No breach when insufficient samples in bin."""
        for i in range(5):
            tid = record_prediction("vrp", 0.9, "0.90-1.00", db_path=temp_db)
            resolve_outcome(tid, 0, db_path=temp_db)

        result = check_ece(sleeve="vrp", min_samples=50, db_path=temp_db)
        assert result["is_breached"] is False  # N < 50

    def test_resolve_invalid_outcome(self, temp_db):
        """Invalid outcome value should raise."""
        tid = record_prediction("kronos", 0.75, "0.70-0.80", db_path=temp_db)
        with pytest.raises(ValueError, match="0 or 1"):
            resolve_outcome(tid, 2, db_path=temp_db)
