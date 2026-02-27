import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

def now_et() -> datetime:
    return datetime.now(ZoneInfo("America/New_York"))

db_path = "backend/artifacts/orchestrator_state/state.sqlite3"
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute("SELECT job_name, last_success_at FROM jobs WHERE job_name IN ('data_refresh_intraday', 'signals_generate_selector', 'fills_reconcile') ORDER BY started_at DESC")
rows = c.fetchall()
print("raw DB entries:")
for r in rows:
    print(r)

now = now_et()
print(f"now_et() = {now}, utc = {now.astimezone(timezone.utc)}")

for r in rows:
    job = r[0]
    last_ok = r[1]
    if last_ok:
        last_ok_dt = datetime.fromisoformat(last_ok)
        elapsed = (now.astimezone(timezone.utc) - last_ok_dt.astimezone(timezone.utc)).total_seconds()
        print(f"job={job} last_ok={last_ok} elapsed={elapsed}")

conn.close()
