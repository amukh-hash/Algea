"""Fetch the execution page HTML and check for errors."""
import urllib.request

# 1. Check backend API
try:
    r = urllib.request.urlopen("http://localhost:8000/api/telemetry/runs?q=Sleeve&limit=5", timeout=5)
    print("Backend API:", r.status, "OK")
    print("  Response:", r.read().decode()[:200])
except Exception as e:
    print("Backend API FAILED:", e)

print()

# 2. Check frontend page
try:
    r = urllib.request.urlopen("http://localhost:3000/execution", timeout=10)
    html = r.read().decode()
    print("Frontend page:", r.status, f"({len(html)} bytes)")
    # Check for error indicators
    if "Application Crash" in html:
        print("  ERROR: Error boundary triggered!")
    if "error" in html.lower():
        # Find error context
        idx = html.lower().find("error")
        print(f"  Found 'error' at position {idx}:", html[max(0,idx-50):idx+100])
    if "Trading Ops" in html:
        print("  Page header rendered OK")
    if "__next" in html:
        print("  Next.js hydration scripts present")
    # Count script tags
    import re
    scripts = re.findall(r'<script[^>]*src="([^"]*)"', html)
    print(f"  {len(scripts)} script tags found")
    for s in scripts[:5]:
        print(f"    {s}")
except Exception as e:
    print("Frontend FAILED:", e)
