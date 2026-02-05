
import pandas as pd
import pandas_market_calendars as mcal
from typing import List

# Singleton Calendar
_NYSE = mcal.get_calendar('NYSE')

def get_trading_days(start_date, end_date) -> List[pd.Timestamp]:
    """
    Returns list of trading days (NYSE) between start and end (inclusive).
    """
    schedule = _NYSE.schedule(start_date=start_date, end_date=end_date)
    return schedule.index.tolist()

def shift_trading_days(date: pd.Timestamp, n: int) -> pd.Timestamp:
    """
    Shifts a date by n trading days. (+n for future, -n for past).
    Uses a broad window to ensure coverage.
    """
    if n == 0:
        return date
        
    start_search = date - pd.Timedelta(days=n*3 + 10) if n < 0 else date
    end_search = date + pd.Timedelta(days=n*3 + 10) if n > 0 else date
    
    # Ensure window covers
    schedule = _NYSE.schedule(start_date=start_search, end_date=end_search)
    dates = schedule.index
    
    # Find pos
    if date not in dates:
        # If date is not a trading day, roll forward/backward?
        # Standard: run from 'asof' (usually trading day).
        # If not found, roll forward to finding insertion point
        loc = dates.searchsorted(date)
    else:
        loc = dates.get_loc(date)
        
    new_loc = loc + n
    
    if new_loc < 0 or new_loc >= len(dates):
        raise ValueError(f"Shift out of bounds. Window too small? {date}, {n}")
        
    return dates[new_loc]
