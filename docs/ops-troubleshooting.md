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

---

## Lint Policy

### CI — Strict Lint

In CI pipelines (GitHub Actions, etc.) always use:

```bash
cd frontend
npm ci
npm run lint:ci        # next lint --max-warnings=0
```

`lint:ci` runs ESLint through Next.js and **fails the build on any warning or error**. This is the authoritative lint gate.

### Restricted / Sandboxed Environments

Some environments (air-gapped hosts, sandboxed build agents) cannot reach the npm registry to install devDependencies. For these, use:

```bash
cd frontend
npm run lint:maybe
```

`lint:maybe` checks whether `eslint` and `eslint-config-next` are installed:

- **Both present** → delegates to `lint:ci` and propagates its exit code.
- **One or both missing** → prints which dependency is absent and exits 0 (non-blocking).

> [!IMPORTANT]
> `lint:maybe` should never be used as the primary CI gate. Its skip behavior is intentional only for environments where `npm install` is impossible.

### Interpreting "lint skipped" Messages

If you see output like:

```
[lint:maybe] Skipping lint — missing devDependencies: eslint, eslint-config-next.
```

This means the environment lacks the listed packages. To resolve:

1. Ensure npm registry access (see `.npmrc` below).
2. Run `npm install` (or `npm ci`) in `frontend/`.
3. Re-run the lint command.

### Private Registry (`.npmrc`)

If your environment uses a private npm registry, create or update `frontend/.npmrc`:

```ini
registry=https://your-registry.example.com/
//your-registry.example.com/:_authToken=${NPM_TOKEN}
```

### Local Verification

```powershell
cd frontend
npm install
npm run lint:ci          # must pass with zero warnings
npm run lint:maybe       # should delegate to lint:ci successfully
npm run test
npx tsc --noEmit         # type-check
```

### Files to Commit

After making lint tooling changes, ensure these are committed together:

- `frontend/package.json`
- `frontend/package-lock.json`
- `frontend/scripts/lint-maybe.mjs`
- `docs/ops-troubleshooting.md`
