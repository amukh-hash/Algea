"""00_init_state.py — Idempotent legacy state purge + canonical schema init.

This script performs a destructive teardown of all legacy artifacts from the
previous SVD/Chronos-2 pipeline, then initialises the new canonical database
tables required by the refactored LiveGuard, ECE tracker, and PatchTST/SMoE
architectures.

Usage:
    python backend/scripts/00_init_state.py          # interactive confirmation
    python backend/scripts/00_init_state.py --force   # skip confirmation
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
from pathlib import Path

# ── Resolve project root ────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = _SCRIPT_DIR.parent.parent  # backend/scripts -> backend -> project
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("init_state")

# ── Paths ────────────────────────────────────────────────────────────────
STATE_DB = PROJECT_ROOT / "backend" / "artifacts" / "orchestrator_state" / "state.sqlite3"
DATA_CANONICAL = PROJECT_ROOT / "backend" / "data_canonical"
OHLCV_DIR = DATA_CANONICAL / "ohlcv_adj"
FEATURES_DIR = DATA_CANONICAL / "features"
ARTIFACTS_ROOT = PROJECT_ROOT / "backend" / "artifacts"

# File extensions to purge from artifacts
WEIGHT_EXTS = {".pt", ".safetensors", ".pth", ".ckpt"}
DATA_EXTS = {".parquet"}


# ═══════════════════════════════════════════════════════════════════════
# 1. File System Purge
# ═══════════════════════════════════════════════════════════════════════

def _purge_directory(root: Path, extensions: set[str], *, label: str) -> int:
    """Recursively delete files matching *extensions* under *root*.

    Returns the count of deleted files.
    """
    if not root.exists():
        logger.info("SKIP  %s — directory does not exist: %s", label, root)
        return 0

    count = 0
    for fp in root.rglob("*"):
        if fp.is_file() and fp.suffix.lower() in extensions:
            fp.unlink()
            count += 1
            logger.debug("DEL   %s", fp.relative_to(PROJECT_ROOT))
    logger.info("PURGE %s — deleted %d file(s) from %s", label, count, root)
    return count


def purge_legacy_files() -> dict[str, int]:
    """Remove all legacy data and model weight files."""
    results: dict[str, int] = {}

    # Parquet data in canonical directories
    results["ohlcv_adj"] = _purge_directory(
        OHLCV_DIR, DATA_EXTS, label="OHLCV Adjusted"
    )
    results["features"] = _purge_directory(
        FEATURES_DIR, DATA_EXTS, label="Features"
    )

    # Model weights and checkpoints across all artifacts
    results["model_weights"] = _purge_directory(
        ARTIFACTS_ROOT, WEIGHT_EXTS, label="Model Weights"
    )

    return results


# ═══════════════════════════════════════════════════════════════════════
# 2. Database Purge & Schema Recreation
# ═══════════════════════════════════════════════════════════════════════

# Legacy tables that must be dropped unconditionally
_LEGACY_TABLES = [
    "confidence_bins",
    "legacy_hit_rates",
    "orchestrator_locks",
    "chronos2_state",
    "svd_components",
]

# Table DDL for the new system
_SCHEMA_DDL = """
-- ── LiveGuard baselines (MMD kernel embeddings) ─────────────────────
CREATE TABLE IF NOT EXISTS liveguard_baselines (
    sleeve        VARCHAR(50) PRIMARY KEY,
    mmd_kernel_mean    BLOB NOT NULL,       -- serialised np.float32 tensor
    feature_mean       BLOB NOT NULL,       -- serialised per-feature means
    feature_variance   BLOB NOT NULL,       -- serialised per-feature variances
    covariance_matrix  BLOB,                -- optional serialised covariance
    sample_count       INTEGER NOT NULL DEFAULT 0,
    updated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ── ECE calibration tracking ────────────────────────────────────────
CREATE TABLE IF NOT EXISTS ece_tracking (
    trade_id              TEXT PRIMARY KEY,
    sleeve                VARCHAR(50) NOT NULL,
    confidence_bin        VARCHAR(20) NOT NULL,  -- e.g. '0.80-0.90'
    predicted_probability REAL NOT NULL,
    actual_outcome        INTEGER,               -- 1 (Win), 0 (Loss), NULL (Pending)
    recorded_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    prediction_date       TEXT,                   -- ISO date of the prediction
    predicted_direction   REAL,                   -- +1 bullish, -1 bearish
    t0_price              REAL,                   -- price at prediction time
    symbol                TEXT,                   -- ticker symbol
    t1_price              REAL,                   -- T+1 settlement price
    resolved_at           TIMESTAMP,              -- when outcome was resolved
    cum_split_factor      REAL DEFAULT 1.0,       -- corporate action split adj
    cum_dividend_factor   REAL DEFAULT 1.0        -- corporate action dividend adj
);
CREATE INDEX IF NOT EXISTS idx_ece_sleeve_bin
    ON ece_tracking(sleeve, confidence_bin);
CREATE INDEX IF NOT EXISTS idx_ece_outcome_pending
    ON ece_tracking(actual_outcome) WHERE actual_outcome IS NULL;

-- ── Orchestrator FSM state ──────────────────────────────────────────
CREATE TABLE IF NOT EXISTS dag_state (
    run_id          TEXT PRIMARY KEY,
    current_state   VARCHAR(30) NOT NULL DEFAULT 'PENDING',
    entered_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    halt_reason     TEXT,
    meta_json       TEXT
);
"""


def purge_and_init_database() -> None:
    """Drop legacy tables, apply WAL mode, and create new schema tables."""
    if not STATE_DB.parent.exists():
        STATE_DB.parent.mkdir(parents=True, exist_ok=True)
        logger.info("MKDIR %s", STATE_DB.parent)

    conn = sqlite3.connect(STATE_DB, timeout=30)
    try:
        # Enforce WAL and pragmas
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        logger.info("DB    WAL mode enabled on %s", STATE_DB)

        # Drop legacy tables
        for table in _LEGACY_TABLES:
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            logger.info("DROP  table '%s' (if existed)", table)

        # Drop and recreate new tables to ensure clean state
        for new_table in ("liveguard_baselines", "ece_tracking", "dag_state"):
            conn.execute(f"DROP TABLE IF EXISTS {new_table}")
            logger.info("DROP  table '%s' (clean slate)", new_table)

        # Create new schema
        conn.executescript(_SCHEMA_DDL)
        logger.info("DDL   New schema tables created: liveguard_baselines, ece_tracking, dag_state")

        conn.commit()
    finally:
        conn.close()


# ═══════════════════════════════════════════════════════════════════════
# 3. Entrypoint
# ═══════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Purge legacy state and init canonical schema.")
    parser.add_argument(
        "--force", action="store_true",
        help="Skip interactive confirmation prompt.",
    )
    args = parser.parse_args()

    if not args.force:
        print(
            "\n⚠️  WARNING: This script will PERMANENTLY DELETE:\n"
            f"  • All .parquet files in {OHLCV_DIR}\n"
            f"  • All .parquet files in {FEATURES_DIR}\n"
            f"  • All model weights (.pt, .safetensors, .pth, .ckpt) in {ARTIFACTS_ROOT}\n"
            f"  • Legacy database tables in {STATE_DB}\n"
        )
        answer = input("Type 'PURGE' to confirm: ").strip()
        if answer != "PURGE":
            logger.info("ABORT — cancelled by user")
            sys.exit(1)

    logger.info("=" * 72)
    logger.info("PHASE 1: Purging legacy files")
    logger.info("=" * 72)
    results = purge_legacy_files()
    total_files = sum(results.values())
    logger.info("TOTAL %d file(s) deleted across %d categories", total_files, len(results))

    logger.info("=" * 72)
    logger.info("PHASE 2: Database purge & schema initialization")
    logger.info("=" * 72)
    purge_and_init_database()

    logger.info("=" * 72)
    logger.info("DONE  Clean slate established. System ready for canonical re-ingestion.")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()
