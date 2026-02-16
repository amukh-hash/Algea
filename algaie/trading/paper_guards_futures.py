"""Paper-trading safety guards for futures execution.

Guards are applied *before* calling ``submit_orders`` to enforce risk
limits, regime constraints, and operational caps.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from algaie.trading.orders import OrderIntent

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PaperGuardConfig:
    """Safety guard parameters for paper-trading futures."""

    max_orders_per_day: int = 20
    max_contracts_per_order: int = 5
    max_contracts_per_instrument: int = 10
    max_gross_notional: float = 5_000_000.0
    roll_window_block: bool = True
    allow_flatten_bypass: bool = True  # never block flatten/close orders


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


@dataclass
class GuardResult:
    """Result of guard checks on a batch of order intents."""

    passed: bool
    violations: List[str] = field(default_factory=list)
    filtered_intents: List[OrderIntent] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "num_filtered_intents": len(self.filtered_intents),
        }


# ---------------------------------------------------------------------------
# Regime enum (avoid circular import)
# ---------------------------------------------------------------------------

_CRASH_RISK = "CRASH_RISK"
_CAUTION = "CAUTION"


# ---------------------------------------------------------------------------
# Guard checks
# ---------------------------------------------------------------------------


def apply_paper_guards(
    intents: List[OrderIntent],
    config: PaperGuardConfig,
    regime: str = "NORMAL_CARRY",
    multipliers: Optional[Dict[str, float]] = None,
    reference_prices: Optional[Dict[str, float]] = None,
    is_flatten: bool = False,
    is_roll_window: bool = False,
) -> GuardResult:
    """Apply all safety guards to a batch of order intents.

    Parameters
    ----------
    intents
        Order intents to filter.
    config
        Guard configuration.
    regime
        Current volatility regime string (``NORMAL_CARRY``, ``CAUTION``, ``CRASH_RISK``).
    multipliers
        Contract multiplier per root (for notional check).
    reference_prices
        Last known price per root (for notional check).
    is_flatten
        If True, these are end-of-day flatten orders (bypass most guards).
    is_roll_window
        If True, we are in the roll window (entry orders blocked if configured).

    Returns
    -------
    GuardResult
        Filtered intents + any violations found.
    """
    violations: List[str] = []
    filtered: List[OrderIntent] = []

    # -----------------------------------------------------------------------
    # Flatten bypass: never block flatten orders
    # -----------------------------------------------------------------------
    if is_flatten and config.allow_flatten_bypass:
        return GuardResult(passed=True, violations=[], filtered_intents=list(intents))

    # -----------------------------------------------------------------------
    # CRASH_RISK: block all entry orders
    # -----------------------------------------------------------------------
    if regime == _CRASH_RISK:
        violations.append(f"CRASH_RISK regime active — all {len(intents)} entry orders blocked")
        logger.warning("CRASH_RISK: blocking %d entry orders", len(intents))
        return GuardResult(passed=False, violations=violations, filtered_intents=[])

    # -----------------------------------------------------------------------
    # Roll window block
    # -----------------------------------------------------------------------
    if is_roll_window and config.roll_window_block:
        violations.append("Roll window active — entry orders blocked")
        logger.warning("Roll window: blocking %d entry orders", len(intents))
        return GuardResult(passed=False, violations=violations, filtered_intents=[])

    # -----------------------------------------------------------------------
    # Order count cap
    # -----------------------------------------------------------------------
    if len(intents) > config.max_orders_per_day:
        violations.append(
            f"Exceeded max_orders_per_day: {len(intents)} > {config.max_orders_per_day}"
        )

    # -----------------------------------------------------------------------
    # Per-order and per-instrument checks
    # -----------------------------------------------------------------------
    instrument_qty: Dict[str, float] = {}

    for intent in intents:
        # Parse root from ticker
        root = _extract_root(intent.ticker)
        abs_qty = abs(intent.quantity)
        intent_ok = True

        # Per-order qty cap
        if abs_qty > config.max_contracts_per_order:
            violations.append(
                f"{intent.ticker}: qty {abs_qty} > max_contracts_per_order {config.max_contracts_per_order}"
            )
            intent_ok = False

        # Per-instrument qty cap
        instrument_qty[root] = instrument_qty.get(root, 0.0) + abs_qty
        if instrument_qty[root] > config.max_contracts_per_instrument:
            violations.append(
                f"{root}: cumulative qty {instrument_qty[root]} > max_contracts_per_instrument "
                f"{config.max_contracts_per_instrument}"
            )
            intent_ok = False

        if intent_ok:
            filtered.append(intent)

    # -----------------------------------------------------------------------
    # Gross notional cap
    # -----------------------------------------------------------------------
    if multipliers and reference_prices:
        gross_notional = 0.0
        for intent in filtered:
            root = _extract_root(intent.ticker)
            mult = multipliers.get(root, 1.0)
            price = reference_prices.get(root, 0.0)
            gross_notional += abs(intent.quantity) * mult * price

        if gross_notional > config.max_gross_notional:
            violations.append(
                f"Gross notional ${gross_notional:,.0f} > cap ${config.max_gross_notional:,.0f}"
            )

    # -----------------------------------------------------------------------
    # CAUTION: log scaling reminder (checked but not blocking)
    # -----------------------------------------------------------------------
    if regime == _CAUTION:
        logger.info("CAUTION regime: ensure caution_scale was applied in sleeve before guards")

    passed = len(violations) == 0
    if not passed:
        logger.warning("Paper guard violations: %s", violations)

    return GuardResult(passed=passed, violations=violations, filtered_intents=filtered)


def _extract_root(ticker: str) -> str:
    """Extract root symbol from an active contract ticker like ESH26."""
    # Root is the alphabetic prefix (2-4 chars) before the month code
    import re
    m = re.match(r"^([A-Z]{2,4})[FGHJKMNQUVXZ]", ticker)
    if m:
        return m.group(1)
    return ticker  # fallback: use full ticker
