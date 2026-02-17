"""Check if the backend API is reachable and returns the family run."""
import urllib.request
import json
import sys

try:
    r = urllib.request.urlopen("http://localhost:8000/api/telemetry/runs?q=Sleeve&limit=5", timeout=3)
    d = json.loads(r.read())
    print(f"API OK: {d['total']} runs matching 'Sleeve'")
    for item in d["items"]:
        tags = item.get("tags", [])
        print(f"  {item['run_id'][:12]}  {item['name'][:50]:50s}  tags={tags}")
except Exception as e:
    print(f"API FAILED: {e}")
    sys.exit(1)
