from enum import Enum
from datetime import timedelta
import pandas as pd
from backend.app.data import calendar

class TradingWindow(Enum):
    CLOSED = "CLOSED"
    PRE_OPEN = "PRE_OPEN" # 30 mins before open
    EARLY_ENTRY = "EARLY_ENTRY" # Open+15 to Open+90
    MID_DAY = "MID_DAY" # Management
    LATE_ADJUST = "LATE_ADJUST" # Close-30 to Close
    POST_CLOSE = "POST_CLOSE"

class SwingScheduler:
    def __init__(self):
        pass

    def get_window(self, ts: pd.Timestamp) -> TradingWindow:
        cal = calendar.get_calendar()
        
        # Ensure UTC
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
            
        if not cal.is_open_on_minute(ts):
            # Check if pre-open?
            next_open = cal.next_open(ts)
            diff = next_open - ts
            if timedelta(minutes=0) < diff <= timedelta(minutes=30):
                return TradingWindow.PRE_OPEN
            return TradingWindow.CLOSED
            
        # It is open. Find session open/close times.
        # session = cal.minute_to_session(ts)
        # open_time = cal.session_open(session)
        # close_time = cal.session_close(session)
        
        # Optimized: get open/close for the *current* open session.
        # But exchange_calendars doesn't have "current_session_open/close" easily for a timestamp inside.
        # We can use `previous_open` (which is this session's open) and `next_close` (this session's close).
        
        open_time = cal.previous_open(ts)
        close_time = cal.next_close(ts)
        
        # Check intervals
        # Early Entry: Open+15m to Open+90m (1.5h)
        if open_time + timedelta(minutes=15) <= ts <= open_time + timedelta(minutes=90):
            return TradingWindow.EARLY_ENTRY
            
        # Late Adjust: Close-30m to Close
        if ts >= close_time - timedelta(minutes=30):
            return TradingWindow.LATE_ADJUST
            
        return TradingWindow.MID_DAY

    def get_allowed_actions(self, window: TradingWindow) -> list:
        if window == TradingWindow.EARLY_ENTRY:
            return ["BUY", "SELL", "HOLD", "REDUCE"]
        elif window == TradingWindow.LATE_ADJUST:
            return ["REDUCE", "LIQUIDATE", "HOLD"] # Maybe late entry?
        elif window == TradingWindow.MID_DAY:
            return ["REDUCE", "HOLD"] # No new entries mid-day for swing?
        return []
