import sqlite3

db_path = "backend/artifacts/orchestrator_state/state.sqlite3"
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute("SELECT error_summary, stderr_path FROM jobs WHERE job_name='signals_generate_selector' ORDER BY started_at DESC LIMIT 1")
row = c.fetchone()
print(row)
conn.close()

if row and row[1]:
    with open(row[1], 'r') as f:
        print(f.read())
