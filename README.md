# Trading Ops Telemetry + UI

## One-command local dev

```bash
python -m uvicorn backend.app.api.main:app --reload &
python backend/scripts/dev_emit_fake_telemetry.py --iterations 500 --sleep 0.5 &
cd frontend && npm install && npm run dev
```

## Backend

- FastAPI app: `backend.app.api.main:app`
- Telemetry routes mounted at `/api/telemetry`
- Storage defaults to SQLite (`backend/telemetry.db`) with Postgres-first SQL migration in `backend/app/telemetry/migrations/001_init_telemetry.sql`

## Frontend

- Next.js app router UI
- Routes:
  - `/execution`
  - `/research`
  - `/runs/[runId]`
  - `/compare?runIds=...`

## Simulation

`backend/scripts/dev_emit_fake_telemetry.py` emits:

- 3 sleeve runs (A/B/C)
- 1 training run
- 1 backtest run
- metrics, events, and summary artifacts
