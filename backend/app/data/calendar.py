import exchange_calendars as ecals
import pandas as pd
from datetime import datetime, timedelta
import pytz

# Singleton
_CALENDAR = None

def get_calendar():
    global _CALENDAR
    if _CALENDAR is None:
        _CALENDAR = ecals.get_calendar("XNYS") # NYSE
    return _CALENDAR

def get_next_session_close(ts: pd.Timestamp, n_sessions: int = 1) -> pd.Timestamp:
    """
    Returns the close time of the n-th next trading session.
    If ts is currently in a session, the current session counts as the first one
    IF we are considering 'next close'.
    However, usually for targets:
    - At time T (during session), 1D target is the Close of THIS session.
    - At time T (after close), 1D target is the Close of NEXT session.

    Standard definition for this project (per plan):
    “1D” target = next RTH session close relative to the decision timestamp.

    If we are IN a session, the 'next close' is this session's close.
    If we are AFTER a session (but before next open), the 'next close' is the next session's close.
    """
    cal = get_calendar()

    # Ensure ts is UTC
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    # Get next close
    # exchange_calendars next_close returns the next close AFTER ts.
    # If ts is 10:00 AM, next_close is 4:00 PM today.
    # If ts is 5:00 PM, next_close is 4:00 PM tomorrow.

    # We use next_close from the calendar
    # Note: next_close behavior in exchange_calendars:
    # "The first close after date."

    try:
        next_close_val = cal.next_close(ts)
    except ValueError:
        # Fallback for far future or edge cases, though exchange_calendars handles broad ranges
        # Trying to find the next session manually if next_close fails (unlikely)
        return ts + timedelta(days=1)

    if n_sessions == 1:
        return next_close_val

    # If n_sessions > 1, we need to find subsequent closes.
    # We can use the schedule.
    # Get the session label for the first next_close
    # We use direction='previous' because the close time is inclusive to the session it ends.
    session_label = cal.minute_to_session(next_close_val, direction='previous')

    # Get window starting from that session
    # We need n_sessions - 1 more sessions after the current one
    # So we get n_sessions total starting from session_label

    # schedule expects start and end, or start and count?
    # cal.sessions_window(start, count)

    sessions = cal.sessions_window(session_label, n_sessions)
    last_session = sessions[-1]

    # Get close of that last session
    return cal.schedule.loc[last_session, "close"]

def is_market_open(ts: pd.Timestamp) -> bool:
    cal = get_calendar()
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return cal.is_open(ts)
