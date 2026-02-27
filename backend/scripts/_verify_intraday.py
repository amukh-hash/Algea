import logging
logging.basicConfig(level=logging.INFO)
from backend.app.orchestrator.orchestrator import Orchestrator
from backend.app.orchestrator.calendar import Session

orch = Orchestrator()
print("Run 1 (INTRADAY):")
result1 = orch.run_once(forced_session=Session.INTRADAY, dry_run=True)
print(result1)

import sqlite3
db_path = "backend/artifacts/orchestrator_state/state.sqlite3"
conn = sqlite3.connect(db_path)
c = conn.cursor()
c.execute("SELECT job_name, status FROM jobs WHERE session='intraday' AND asof_date='2026-02-24' AND job_name='signals_generate_selector'")
print("Selector DB Status:")
for row in c.fetchall():
    print(row)
conn.close()
