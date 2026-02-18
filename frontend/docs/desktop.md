# Desktop (Tauri + FastAPI sidecar)

## Architecture
- Tauri shell (`src-tauri/`) supervises `backend_dist/orchestrator.exe`.
- Backend port is dynamically allocated (no hardcoded 8000).
- Frontend resolves API base URL from `invoke("get_backend_base_url")` in desktop mode.

## Commands
- `npm run desktop:dev` (runs Tauri dev)
- `npm run desktop:build` (builds desktop bundle)

## Sidecar lifecycle
- Start: `backend_dist/orchestrator.exe --host 127.0.0.1 --port <dynamic>`
- Readiness: poll `/healthz` up to 15s.
- Shutdown: sidecar killed on app close.

## Manual QA checklist
1. Launch desktop app: backend starts and `/healthz` returns ok.
2. Open `/execution`: streaming updates appear.
3. Click restart backend in Settings: reconnect succeeds.
4. Quit app: sidecar process exits.
5. Relaunch multiple times: no stale fixed port conflicts.
