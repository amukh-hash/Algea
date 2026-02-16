"""IBKR futures contract qualification and caching utilities."""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ib_insync import Contract, Future  # type: ignore[import-untyped]

from .ibkr_client import IbkrClient, QualifiedContract

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Exchange mapping
# ---------------------------------------------------------------------------

EXCHANGE_MAP: Dict[str, str] = {
    # Equity indices
    "ES": "CME",
    "NQ": "CME",
    "RTY": "CME",
    "YM": "ECBOT",
    # Energy
    "CL": "NYMEX",
    # Metals
    "GC": "COMEX",
    "SI": "COMEX",
    "HG": "COMEX",
    # Rates
    "ZN": "ECBOT",
    "ZB": "ECBOT",
    # FX
    "6E": "CME",
    "6J": "CME",
    "6B": "CME",
    "6A": "CME",
    # Micros
    "MES": "CME",
    "MNQ": "CME",
    "MYM": "ECBOT",
    "M2K": "CME",
}

# ---------------------------------------------------------------------------
# Month code helpers (shared with sleeves.cooc_reversal_futures.roll)
# ---------------------------------------------------------------------------

_MONTH_CODE_MAP: Dict[str, int] = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

# Compiled regex patterns for active contract symbol parsing
# Pattern 1: ESH26  (root + code + 2-digit year)
_RE_SHORT = re.compile(r"^([A-Z0-9]{2,4})([FGHJKMNQUVXZ])(\d{2})$")
# Pattern 2: ESU2026 (root + code + 4-digit year)
_RE_LONG = re.compile(r"^([A-Z0-9]{2,4})([FGHJKMNQUVXZ])(\d{4})$")
# Pattern 3: ES-202609 (root + dash + YYYYMM)
_RE_DASH = re.compile(r"^([A-Z0-9]{2,4})-(\d{6})$")


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_active_contract_symbol(symbol: str) -> Tuple[str, str]:
    """Parse an active contract symbol into ``(root, expiry_yyyymm)``.

    Supported formats:

    * ``ESH26``   → ``("ES", "202603")``  (2-digit year, assumes 2000s)
    * ``ESU2026`` → ``("ES", "202609")``
    * ``ES-202609`` → ``("ES", "202609")``

    Raises
    ------
    ValueError
        If the symbol format is unrecognised.
    """
    # Try short format first (most common from roll.py)
    m = _RE_SHORT.match(symbol)
    if m:
        root = m.group(1)
        code = m.group(2)
        year_2d = int(m.group(3))
        month = _MONTH_CODE_MAP[code]
        year = 2000 + year_2d
        return root, f"{year}{month:02d}"

    # Long format
    m = _RE_LONG.match(symbol)
    if m:
        root = m.group(1)
        code = m.group(2)
        year = int(m.group(3))
        month = _MONTH_CODE_MAP[code]
        return root, f"{year}{month:02d}"

    # Dash format
    m = _RE_DASH.match(symbol)
    if m:
        root = m.group(1)
        yyyymm = m.group(2)
        return root, yyyymm

    raise ValueError(
        f"Cannot parse active contract symbol '{symbol}'. "
        f"Expected formats: ESH26, ESU2026, or ES-202609."
    )


# ---------------------------------------------------------------------------
# Contract building
# ---------------------------------------------------------------------------


def build_future_contract(
    root: str,
    expiry_yyyymm: str,
    exchange: Optional[str] = None,
    currency: str = "USD",
    include_expired: bool = False,
) -> Future:
    """Build an ``ib_insync.Future`` for qualification.

    Parameters
    ----------
    root
        Root symbol (e.g. ``"ES"``).
    expiry_yyyymm
        Expiry in ``YYYYMM`` format (e.g. ``"202603"``).
    exchange
        Exchange override.  If ``None``, looked up from :data:`EXCHANGE_MAP`.
    currency
        Contract currency (default ``"USD"``).
    include_expired
        If True, sets ``includeExpired=True`` so IBKR can qualify
        and return data for expired contracts.
    """
    if exchange is None:
        exchange = EXCHANGE_MAP.get(root)
        if exchange is None:
            raise ValueError(
                f"No exchange mapping for root '{root}'. "
                f"Available: {sorted(EXCHANGE_MAP.keys())}"
            )
    return Future(
        symbol=root,
        lastTradeDateOrContractMonth=expiry_yyyymm,
        exchange=exchange,
        currency=currency,
        includeExpired=include_expired,
    )


# ---------------------------------------------------------------------------
# Disk cache for qualified contracts
# ---------------------------------------------------------------------------

_DEFAULT_CACHE_PATH = Path("artifacts/broker_cache/ibkr_contracts.json")


def _cache_key(root: str, expiry: str, exchange: str, currency: str) -> str:
    return f"{root}|{expiry}|{exchange}|{currency}"


def _load_cache(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt contract cache at %s — starting fresh", cache_path)
    return {}


def _save_cache(cache: Dict[str, Dict[str, Any]], cache_path: Path) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(cache, indent=2, sort_keys=True), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Qualification
# ---------------------------------------------------------------------------


def qualify_future(
    client: IbkrClient,
    contract: Future,
    cache_path: Path = _DEFAULT_CACHE_PATH,
) -> QualifiedContract:
    """Qualify a futures contract via IBKR, with disk caching.

    Parameters
    ----------
    client
        Connected :class:`IbkrClient`.
    contract
        Unqualified ``ib_insync.Future``.
    cache_path
        Path to the JSON cache file.

    Returns
    -------
    QualifiedContract
        Qualified contract with conId, localSymbol, etc.
    """
    root = contract.symbol
    expiry = contract.lastTradeDateOrContractMonth
    exchange = contract.exchange
    currency = contract.currency
    key = _cache_key(root, expiry, exchange, currency)

    # Check cache
    cache = _load_cache(cache_path)
    if key in cache:
        logger.debug("Contract cache hit: %s", key)
        return QualifiedContract(**cache[key])

    # Qualify via IBKR
    logger.info("Qualifying contract: %s %s %s %s", root, expiry, exchange, currency)
    qualified = client.qualify_contracts(contract)
    if not qualified or qualified[0].conId == 0:
        raise RuntimeError(
            f"Failed to qualify contract {root} {expiry} on {exchange}. "
            f"Check symbol, exchange, and that TWS/Gateway is connected."
        )

    q = qualified[0]
    result = QualifiedContract(
        con_id=q.conId,
        symbol=q.symbol,
        local_symbol=q.localSymbol,
        exchange=q.exchange,
        currency=q.currency,
        multiplier=str(q.multiplier),
        expiry=q.lastTradeDateOrContractMonth,
        trading_class=q.tradingClass or "",
    )

    # Persist to cache
    cache[key] = result.to_dict()
    _save_cache(cache, cache_path)
    logger.info("Qualified: %s → conId=%d localSymbol=%s", key, result.con_id, result.local_symbol)

    return result


def qualify_active_contract(
    client: IbkrClient,
    active_contract_symbol: str,
    cache_path: Path = _DEFAULT_CACHE_PATH,
) -> QualifiedContract:
    """Parse an active contract symbol, build, and qualify in one call.

    Parameters
    ----------
    client
        Connected :class:`IbkrClient`.
    active_contract_symbol
        e.g. ``"ESH26"``, ``"NQU2026"``, or ``"YM-202603"``.
    """
    root, expiry = parse_active_contract_symbol(active_contract_symbol)
    contract = build_future_contract(root, expiry)
    return qualify_future(client, contract, cache_path=cache_path)
