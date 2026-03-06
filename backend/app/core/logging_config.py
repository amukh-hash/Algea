"""Centralized logging configuration for the Algaie backend.

Day-2 Mitigation — Log Exhaustion & SQLITE_FULL:
    Standard ``logging.FileHandler`` implementations will write tens of
    gigabytes of text over a 30-day continuous run.  Once the host NVMe
    drive hits 100% capacity, SQLite throws ``OperationalError: database
    or disk is full``, crashing the orchestrator and corrupting the
    LiveGuard Risk Engine mid-execution.

    This module replaces all file handlers with strict
    ``TimedRotatingFileHandler`` classes that rotate at midnight UTC
    and physically delete logs older than 7 days.
"""
from __future__ import annotations

import logging
import os
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

_DEFAULT_LOG_DIR = Path(__file__).resolve().parents[3] / "logs"
_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s [%(process)d] %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def configure_logging(
    *,
    log_dir: Path | str | None = None,
    level: int = logging.INFO,
    backup_count: int = 7,
    console: bool = True,
) -> None:
    """Configure root logger with rotating file + optional console handlers.

    Parameters
    ----------
    log_dir : Path | str | None
        Directory for log files.  Falls back to ``LOG_DIR`` env var,
        then ``backend/logs/``.
    level : int
        Logging level (default: INFO).
    backup_count : int
        Number of rotated log files to keep (default: 7 days).
    console : bool
        Whether to also attach a console (stderr) handler (default: True).
    """
    resolved_dir = Path(log_dir or os.getenv("LOG_DIR", str(_DEFAULT_LOG_DIR)))
    resolved_dir.mkdir(parents=True, exist_ok=True)

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any existing handlers to prevent duplicate output
    root.handlers.clear()

    # ── Rotating file handler ───────────────────────────────────────────
    file_handler = TimedRotatingFileHandler(
        filename=str(resolved_dir / "algaie.log"),
        when="midnight",
        interval=1,
        backupCount=backup_count,
        utc=True,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
    root.addHandler(file_handler)

    # ── Console handler ─────────────────────────────────────────────────
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATE_FORMAT))
        root.addHandler(console_handler)
