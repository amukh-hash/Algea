# ALGAIE Control Room (MVP)

## Backend (FastAPI)

```bash
uvicorn backend.app.api.main:app --host 0.0.0.0 --port 8000
```

### Runs storage

All runs are stored under:

```
backend/data/runs/
```

### Useful endpoints

```bash
# list runs
curl http://localhost:8000/control-room/runs

# fetch a run
curl http://localhost:8000/control-room/runs/RUN-YYYY-MM-DD-001

# events
curl http://localhost:8000/control-room/runs/RUN-YYYY-MM-DD-001/events?tail=200
```

## Frontend (Vite + React)

```bash
cd frontend
npm install
npm run dev
```

The frontend proxies API requests to `http://localhost:8000` by default.
