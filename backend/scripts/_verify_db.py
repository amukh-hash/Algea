import sqlite3, json

c = sqlite3.connect("backend/telemetry.db")
c.row_factory = sqlite3.Row

r = c.execute("SELECT run_id, name, tags, meta FROM runs WHERE tags LIKE '%family%' LIMIT 1").fetchone()
if not r:
    print("ERROR: No family run found!")
    exit(1)

print("Run:", r["name"])
print("ID: ", r["run_id"])
print("Tags:", r["tags"])
meta = json.loads(r["meta"])
print("Members:", meta.get("family_members"))
print()

pts = c.execute(
    "SELECT key, ts, value FROM metrics WHERE run_id=? ORDER BY key, ts",
    (r["run_id"],),
).fetchall()
print(f"Total metric points: {len(pts)}")
for p in pts:
    print(f"  {p['key']:40s}  {p['ts'][:19]}  {p['value']}")

print()
evts = c.execute(
    "SELECT level, type, message FROM events WHERE run_id=? ORDER BY ts",
    (r["run_id"],),
).fetchall()
print(f"Total events: {len(evts)}")
for e in evts:
    print(f"  [{e['level']}] {e['type']}: {e['message'][:80]}")
