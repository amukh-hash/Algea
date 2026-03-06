"""Tests for the centralized logging configuration.

Verifies that configure_logging correctly sets up TimedRotatingFileHandler
with the expected rotation policy and log file paths.
"""
from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _clean_root_logger():
    """Reset root logger handlers before and after each test."""
    root = logging.getLogger()
    original_handlers = root.handlers[:]
    original_level = root.level
    yield
    root.handlers = original_handlers
    root.level = original_level


class TestConfigureLogging:

    def test_file_handler_attached(self, tmp_path):
        """configure_logging should attach a TimedRotatingFileHandler."""
        from backend.app.core.logging_config import configure_logging
        configure_logging(log_dir=tmp_path, console=False)

        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert len(file_handlers) == 1

    def test_backup_count_is_seven(self, tmp_path):
        """Default backupCount should be 7 (7-day retention)."""
        from backend.app.core.logging_config import configure_logging
        configure_logging(log_dir=tmp_path, console=False)

        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert file_handlers[0].backupCount == 7

    def test_custom_backup_count(self, tmp_path):
        """backupCount should honor the backup_count parameter."""
        from backend.app.core.logging_config import configure_logging
        configure_logging(log_dir=tmp_path, console=False, backup_count=14)

        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert file_handlers[0].backupCount == 14

    def test_log_file_created(self, tmp_path):
        """Log file should be created in the specified directory."""
        from backend.app.core.logging_config import configure_logging
        configure_logging(log_dir=tmp_path, console=False)

        log_file = tmp_path / "algaie.log"
        # Write a test message to force file creation
        logging.getLogger("test").info("test message")
        assert log_file.exists()

    def test_utc_rotation(self, tmp_path):
        """Rotation should use UTC time (utc=True)."""
        from backend.app.core.logging_config import configure_logging
        configure_logging(log_dir=tmp_path, console=False)

        root = logging.getLogger()
        file_handlers = [h for h in root.handlers if isinstance(h, TimedRotatingFileHandler)]
        assert file_handlers[0].utc is True

    def test_console_handler_included_by_default(self, tmp_path):
        """Console handler should be present when console=True (default)."""
        from backend.app.core.logging_config import configure_logging
        configure_logging(log_dir=tmp_path, console=True)

        root = logging.getLogger()
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)
                           and not isinstance(h, TimedRotatingFileHandler)]
        assert len(stream_handlers) >= 1

    def test_no_console_handler_when_disabled(self, tmp_path):
        """No console handler when console=False."""
        from backend.app.core.logging_config import configure_logging
        configure_logging(log_dir=tmp_path, console=False)

        root = logging.getLogger()
        stream_handlers = [h for h in root.handlers if isinstance(h, logging.StreamHandler)
                           and not isinstance(h, TimedRotatingFileHandler)]
        assert len(stream_handlers) == 0
