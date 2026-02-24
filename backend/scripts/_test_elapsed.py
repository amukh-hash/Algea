from datetime import datetime, timezone
from zoneinfo import ZoneInfo

def now_et() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))

now = now_et()
last_ok = "2026-02-23T15:44:33-05:00"
last_ok_dt = datetime.fromisoformat(last_ok)

elapsed = (now.astimezone(timezone.utc) - last_ok_dt.astimezone(timezone.utc)).total_seconds()
print("now:", now)
print("last_ok:", last_ok_dt)
print("elapsed:", elapsed)
