from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .contract_master import ContractSpec


# ---------------------------------------------------------------------------
# Month-code → numeric helpers
# ---------------------------------------------------------------------------

_MONTH_CODE_MAP = {
    "F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
    "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12,
}

_MONTH_TO_CODE = {v: k for k, v in _MONTH_CODE_MAP.items()}


def front_contract(month_codes: list[str], ts_date: date) -> str:
    """Deterministic front contract picker shared by backtest/live."""
    idx = (ts_date.month - 1) % len(month_codes)
    return month_codes[idx]


def roll_week_flag(ts_date: date) -> bool:
    """True if the date falls in the typical roll window (8th–14th)."""
    return 8 <= ts_date.day <= 14


def _next_expiry_month(month_codes: list[str], current_month: int) -> int:
    """Return the month number of the next contract expiry on or after *current_month*."""
    months = sorted(_MONTH_CODE_MAP[c] for c in month_codes)
    for m in months:
        if m >= current_month:
            return m
    return months[0]  # wrap to next year


def active_contract_for_day(
    root: str,
    day: date,
    spec: "ContractSpec",
) -> str:
    """Deterministic active contract symbol for *root* on *day*.

    Uses ``front_contract`` and ``roll_week_flag`` to determine which
    contract month is active.  During the roll window the *next*
    quarterly month is returned (simulating roll-forward).

    Returns a string like ``"ESH26"`` (root + month code + 2-digit year).
    """
    codes = list(spec.roll_month_codes)
    base_code = front_contract(codes, day)
    base_month = _MONTH_CODE_MAP[base_code]

    # During roll week, advance to next contract month
    if roll_week_flag(day):
        idx = codes.index(base_code)
        next_idx = (idx + 1) % len(codes)
        code = codes[next_idx]
        month = _MONTH_CODE_MAP[code]
        year = day.year if month > day.month else day.year + 1
    else:
        code = base_code
        month = base_month
        year = day.year if month >= day.month else day.year + 1

    return f"{root}{code}{year % 100:02d}"


def days_to_expiry_estimate(day: date, contract_month: int, contract_year: int) -> int:
    """Rough estimate of calendar days to the 3rd Friday of the expiry month."""
    import calendar
    cal = calendar.Calendar(firstweekday=0)
    fridays = [
        d for d in cal.itermonthdays2(contract_year, contract_month)
        if d[0] != 0 and d[1] == 4  # Friday
    ]
    third_friday = fridays[2][0] if len(fridays) >= 3 else 15
    expiry = date(contract_year, contract_month, third_friday)
    return max(0, (expiry - day).days)
