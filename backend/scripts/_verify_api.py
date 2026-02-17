"""Quick verification script for family run API."""
import urllib.request
import json

# 1. Find family run
r = urllib.request.urlopen("http://localhost:8000/api/telemetry/runs?q=Family")
d = json.loads(r.read())
run = d["items"][0]
print("Run:", run["name"])
print("ID: ", run["run_id"])
print("Tags:", run["tags"])
print("Members:", run["meta"].get("family_members"))
print()

# 2. Query metrics in LW format
keys = "equity,cash,buying_power,sleeve_capital.total,sleeve.selector.intents_count"
url = f"http://localhost:8000/api/telemetry/runs/{run['run_id']}/metrics?keys={keys}&format=lw"
r2 = urllib.request.urlopen(url)
d2 = json.loads(r2.read())
for k, v in d2["series"].items():
    print(f"  {k}: {len(v)} points -> {v}")

# 3. Query events
url3 = f"http://localhost:8000/api/telemetry/runs/{run['run_id']}/events"
r3 = urllib.request.urlopen(url3)
d3 = json.loads(r3.read())
print(f"\nEvents: {len(d3['items'])} total")
for e in d3["items"][:5]:
    print(f"  [{e['level']}] {e['type']}: {e['message'][:80]}")
