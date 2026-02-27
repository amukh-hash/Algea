import sqlite3

db_path = "backend/artifacts/orchestrator_state/state.sqlite3"
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute("PRAGMA table_info(jobs)")
for row in c.fetchall():
    print(row)

conn.close()
