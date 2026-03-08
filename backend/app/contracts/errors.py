"""Structured error taxonomy for Intent Supremacy architecture.

Every failure has an explicit type so the orchestrator can decide
fail-closed vs. degradation based on error category, not message text.
"""
from __future__ import annotations


class SleeveError(Exception):
    """Base class for all sleeve-level errors."""
    pass


class DataUnavailableError(SleeveError):
    """Required data is missing entirely (not stale, just absent)."""
    pass


class DataStaleError(SleeveError):
    """Required data exists but is older than acceptable freshness threshold."""
    pass


class InvalidConfigError(SleeveError):
    """Sleeve configuration is missing, malformed, or out-of-range."""
    pass


class ContractViolationError(SleeveError):
    """Sleeve output does not conform to the canonical contract."""
    pass


class PlanningError(Exception):
    """Error during order plan construction from canonical intents."""
    pass


class RoutingError(Exception):
    """Error during broker order placement/routing."""
    pass
