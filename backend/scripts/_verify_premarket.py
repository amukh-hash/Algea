import logging
logging.basicConfig(level=logging.INFO)
from backend.app.orchestrator.orchestrator import Orchestrator
from backend.app.orchestrator.calendar import Session
orch = Orchestrator()
print("Run 1:")
result1 = orch.run_once(forced_session=Session.PREMARKET, dry_run=True)
print(result1)

print("Run 2:")
result2 = orch.run_once(forced_session=Session.PREMARKET, dry_run=True)
print(result2)

import sqlite3
db_path = "backend/artifacts/orchestrator_state/state.sqlite3"
conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("SELECT job_name, status FROM jobs WHERE session='premarket' AND asof_date='2026-02-24'")
for row in c.fetchall():
    print(row)
conn.close()
