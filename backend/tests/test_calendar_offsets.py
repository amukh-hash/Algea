import pytest
import pandas as pd
from backend.app.data import calendar

def test_calendar_singleton():
    cal1 = calendar.get_calendar()
    cal2 = calendar.get_calendar()
    assert cal1 is cal2

def test_next_session_close_intraday():
    # Thursday, 10 AM ET (market open) -> Expect Thursday Close (4 PM ET)
    # 2023-10-12 is a Thursday.
    # 10 AM ET is 14:00 UTC.
    ts = pd.Timestamp("2023-10-12 14:00:00", tz="UTC")

    # 1D target (next session close)
    # Our definition: "next RTH session close relative to the decision timestamp"
    # If we are IN a session, the "next close" is THIS session's close?
    # Or strict "next"?
    # The plan says: "“1D” target = next RTH session close relative to the decision timestamp"
    # If decision is at 10AM, the session closes at 4PM. That IS the next close event.

    close = calendar.get_next_session_close(ts, n_sessions=1)

    # Expected: 2023-10-12 16:00 ET -> 20:00 UTC
    expected = pd.Timestamp("2023-10-12 20:00:00", tz="UTC")

    assert close == expected

def test_next_session_close_after_market():
    # Thursday, 8 PM ET (after market) -> Expect Friday Close
    # 2023-10-12 20:00 ET (close). Let's say 21:00 UTC (5 PM ET).
    ts = pd.Timestamp("2023-10-12 21:00:00", tz="UTC")

    close = calendar.get_next_session_close(ts, n_sessions=1)

    # Expected: Friday 2023-10-13 16:00 ET -> 20:00 UTC
    expected = pd.Timestamp("2023-10-13 20:00:00", tz="UTC")

    assert close == expected

def test_next_session_close_weekend():
    # Friday Night (after close) -> Expect Monday Close
    ts = pd.Timestamp("2023-10-13 21:00:00", tz="UTC") # Friday

    close = calendar.get_next_session_close(ts, n_sessions=1)

    # Expected: Monday 2023-10-16 16:00 ET -> 20:00 UTC
    expected = pd.Timestamp("2023-10-16 20:00:00", tz="UTC")

    assert close == expected

def test_next_3_session_close():
    # Thursday 10 AM -> 3D target
    # 1D = Thu Close
    # 2D = Fri Close
    # 3D = Mon Close
    ts = pd.Timestamp("2023-10-12 14:00:00", tz="UTC")

    close = calendar.get_next_session_close(ts, n_sessions=3)

    # Expected: Monday 2023-10-16 20:00 UTC
    expected = pd.Timestamp("2023-10-16 20:00:00", tz="UTC")

    assert close == expected
