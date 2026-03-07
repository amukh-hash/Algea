"""Canonical runtime mode enumeration.

All orchestrator, executor, and safety code should compare against
these enum values instead of ad-hoc string matching.

Usage::

    from backend.app.core.runtime_mode import RuntimeMode

    if mode == RuntimeMode.PAPER:
        ...
"""
from __future__ import annotations

from enum import Enum


class RuntimeMode(Enum):
    """Strict execution modes for the Algae trading system."""

    NOOP = "noop"
    STUB = "stub"
    PAPER = "paper"
    LIVE = "ibkr"


class OrchestratorSafetyError(RuntimeError):
    """Raised when a safety-critical operation is invalid for the current runtime mode."""


class ArtifactValidationError(ValueError):
    """Raised when a model artifact fails integrity validation (size, hash, format)."""
