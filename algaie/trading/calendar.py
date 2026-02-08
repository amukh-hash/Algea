from __future__ import annotations

from datetime import date
from typing import Iterable, List

import pandas as pd


def trading_days(start: date, end: date) -> List[date]:
    return [d.date() for d in pd.bdate_range(start=start, end=end)]


def next_trading_day(current: date, calendar: Iterable[date]) -> date | None:
    dates = list(calendar)
    if current not in dates:
        return None
    idx = dates.index(current)
    if idx + 1 >= len(dates):
        return None
    return dates[idx + 1]
