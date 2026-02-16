"""Post-qualification contract spec validation.

After IBKR qualifies a contract, we verify that the returned metadata
matches our ``ContractSpec`` expectations.  Mismatches are logged as
warnings and optionally block promotion (mark pack as ``RESEARCH_ONLY``).
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List, Optional

if TYPE_CHECKING:
    from .contract_master import ContractSpec

logger = logging.getLogger(__name__)

# Month-code → month-number lookup (shared with roll.py / ibkr_contracts.py)
_MONTH_CODE_MAP = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}
_MONTH_TO_CODE = {v: k for k, v in _MONTH_CODE_MAP.items()}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

class SpecCheckResult:
    """Result of a single spec check."""

    __slots__ = ("root", "passed", "warnings")

    def __init__(self, root: str, passed: bool, warnings: List[str]) -> None:
        self.root = root
        self.passed = passed
        self.warnings = warnings

    def to_dict(self) -> Dict[str, object]:
        return {"root": self.root, "passed": self.passed, "warnings": self.warnings}


def validate_qualified_contract(
    spec: "ContractSpec",
    qualified_exchange: str,
    qualified_expiry_yyyymm: str,
    *,
    allow_mismatch: bool = False,
) -> SpecCheckResult:
    """Validate a qualified IBKR contract against its ``ContractSpec``.

    Parameters
    ----------
    spec
        Expected contract spec from the contract master.
    qualified_exchange
        The exchange returned by IBKR qualification.
    qualified_expiry_yyyymm
        The ``lastTradeDateOrContractMonth`` returned by IBKR, as ``YYYYMM``.
    allow_mismatch
        If True, mismatches produce warnings but ``passed`` remains True.
        If False (default), any mismatch causes ``passed=False``.

    Returns
    -------
    SpecCheckResult
        ``passed=True`` means the contract is safe for production promotion.
        ``passed=False`` means the pack should be marked RESEARCH_ONLY.
    """
    warnings: List[str] = []

    # --- Exchange check ---
    expected_exchange = spec.exchange
    if expected_exchange and qualified_exchange != expected_exchange:
        # IBKR sometimes returns slightly different exchange names
        # (e.g. "CBOT" vs "ECBOT"), so we normalise common aliases.
        normalised = _normalise_exchange(qualified_exchange)
        expected_norm = _normalise_exchange(expected_exchange)
        if normalised != expected_norm:
            warnings.append(
                f"Exchange mismatch for {spec.symbol}: "
                f"expected {expected_exchange}, got {qualified_exchange}"
            )

    # --- Month-code check ---
    if len(qualified_expiry_yyyymm) >= 6:
        month_num = int(qualified_expiry_yyyymm[4:6])
        month_code = _MONTH_TO_CODE.get(month_num)
        if month_code and month_code not in spec.roll_month_codes:
            warnings.append(
                f"Month code mismatch for {spec.symbol}: "
                f"qualified month {month_code} ({month_num}) "
                f"not in expected roll cycle {spec.roll_month_codes}"
            )

    passed = True
    if warnings and not allow_mismatch:
        passed = False
        for w in warnings:
            logger.warning("ContractSpec check FAIL: %s", w)
    elif warnings:
        for w in warnings:
            logger.warning("ContractSpec check WARN (allowed): %s", w)

    return SpecCheckResult(root=spec.symbol, passed=passed, warnings=warnings)


def validate_all_specs(
    specs: Dict[str, "ContractSpec"],
    qualified_results: Dict[str, dict],
    *,
    allow_mismatch: bool = False,
) -> List[SpecCheckResult]:
    """Validate a batch of qualified contracts.

    Parameters
    ----------
    specs
        Contract master dict (root → ContractSpec).
    qualified_results
        Dict of root → {"exchange": str, "expiry_yyyymm": str}.
    allow_mismatch
        If True, mismatches produce warnings only.

    Returns
    -------
    List of SpecCheckResult, one per root in qualified_results.
    """
    results: List[SpecCheckResult] = []
    for root, qr in sorted(qualified_results.items()):
        spec = specs.get(root)
        if spec is None:
            results.append(SpecCheckResult(
                root=root, passed=False,
                warnings=[f"Root {root} not in contract master"],
            ))
            continue
        results.append(validate_qualified_contract(
            spec,
            qr.get("exchange", ""),
            qr.get("expiry_yyyymm", ""),
            allow_mismatch=allow_mismatch,
        ))
    return results


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_EXCHANGE_ALIASES = {
    "CBOT": "ECBOT",
    "ECBOT": "ECBOT",
    "CME": "CME",
    "GLOBEX": "CME",
    "NYMEX": "NYMEX",
    "COMEX": "COMEX",
}


def _normalise_exchange(exchange: str) -> str:
    """Map common IBKR exchange variants to a canonical form."""
    return _EXCHANGE_ALIASES.get(exchange.upper(), exchange.upper())
