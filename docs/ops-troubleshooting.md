# Operations Troubleshooting

## Windows Environment Setup

### `HOME` not set (Playwright / Chromium)

Playwright expects `$HOME` (or `USERPROFILE`) to find its browser cache. On Windows, `HOME` is often unset.

**Fix — set `HOME` permanently in PowerShell:**
```powershell
[Environment]::SetEnvironmentVariable("HOME", $env:USERPROFILE, "User")
```

**Or per-session:**
```powershell
$env:HOME = $env:USERPROFILE
```

### `PLAYWRIGHT_BROWSERS_PATH`

To control where Playwright installs Chromium:

```powershell
[Environment]::SetEnvironmentVariable("PLAYWRIGHT_BROWSERS_PATH", "$env:USERPROFILE\.playwright", "User")
npx playwright install chromium
```

---

## Stale Port / Backend Hangs

### Symptoms
- `http://localhost:8000/api/orchestrator/status` hangs indefinitely
- `uvicorn` fails to start with `[Errno 10048]` (port already in use)
- Dashboard shows empty cards with no error messages

### Diagnose

Find what holds port 8000:
```powershell
Get-NetTCPConnection -LocalPort 8000 |
  Select-Object LocalPort, OwningProcess |
  ForEach-Object { Get-Process -Id $_.OwningProcess }
```

### Fix

Kill the stale process:
```powershell
Get-NetTCPConnection -LocalPort 8000 |
  ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

Then restart cleanly:
```powershell
py -m uvicorn backend.app.api.main:app --port 8000
```

---

## Health Check

After starting the backend, verify health:

```
GET http://localhost:8000/healthz
```

Returns:
```json
{
  "ok": true,
  "elapsed_ms": 3.2,
  "checks": {
    "event_loop": { "ok": true, "latency_ms": 0.01 },
    "state_db": { "ok": true, "run_count": 42 },
    "heartbeat": { "ok": true, "age_seconds": 45.3, "session": "intraday" }
  }
}
```

- HTTP 200 = all checks pass
- HTTP 503 = at least one check failed (see `checks` for details)

---

## Request Timeouts

All `/api/orchestrator/*` endpoints enforce a 5-second timeout. If a database query or file read blocks longer than 5s, the backend returns:

```json
HTTP 504
{
  "error": "timeout",
  "detail": "_get_status_sync did not complete within 5.0s"
}
```

The frontend surfaces this as a "Connection Timeout" banner with a Retry button.

---

## Frontend Not Loading

1. Check backend is running: `curl http://localhost:8000/healthz`
2. Check frontend is running: `curl http://localhost:3000`
3. Check CORS: backend must include `http://localhost:3000` in allowed origins (configured in `main.py`)
4. Check browser console for CORS or network errors
