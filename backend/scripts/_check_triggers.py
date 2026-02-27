import sqlite3

db_path = "backend/artifacts/orchestrator_state/state.sqlite3"
conn = sqlite3.connect(db_path)
c = conn.cursor()

c.execute("SELECT name, sql FROM sqlite_master WHERE type='trigger'")
for row in c.fetchall():
    print(row)

conn.close()
