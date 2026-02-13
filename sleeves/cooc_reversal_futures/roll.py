from __future__ import annotations

from datetime import date


def front_contract(month_codes: list[str], ts_date: date) -> str:
    """Deterministic front contract picker shared by backtest/live."""
    idx = (ts_date.month - 1) % len(month_codes)
    return month_codes[idx]


def roll_week_flag(ts_date: date) -> bool:
    return 8 <= ts_date.day <= 14
