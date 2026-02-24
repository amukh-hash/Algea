import sqlite3

db_path = "backend/artifacts/orchestrator_state/state.sqlite3"
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute("SELECT job_name, status, last_success_at, started_at FROM jobs WHERE job_name='signals_generate_selector' ORDER BY started_at DESC LIMIT 10")
for row in c.fetchall():
    print(row)

c.execute("SELECT job_name, status, last_success_at, started_at FROM jobs WHERE job_name='signals_generate_vrp' ORDER BY started_at DESC LIMIT 10")
for row in c.fetchall():
    print(row)

conn.close()
