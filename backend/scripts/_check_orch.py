import sqlite3

db_path = "backend/artifacts/orchestrator_state/state.sqlite3"
conn = sqlite3.connect(db_path)
c = conn.cursor()

print("Recent Jobs:")
c.execute("""
    SELECT asof_date, session, job_name, status, error_summary, started_at 
    FROM jobs 
    ORDER BY started_at DESC 
    LIMIT 30;
""")
for row in c.fetchall():
    print(row)

conn.close()
