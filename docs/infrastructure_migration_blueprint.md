# Infrastructure Migration Blueprint

## 1. SQLite → PostgreSQL

### Affected Modules (7+)

| Module | DB Path | Usage |
|--------|---------|-------|
| `state_store.py` | `state.sqlite3` | Job status, run records |
| `dag_fsm.py` | `state.sqlite3` | DAG state machine |
| `ece_tracker.py` | `state.sqlite3` | ECE calibration tracking |
| `durable_control_state.py` | `state.sqlite3` | App control flags |
| `telemetry/storage.py` | `telemetry.db` | Run telemetry |
| `ml_platform/registry/store.py` | `model_registry.sqlite` | Model artifacts |
| `api/control_routes.py` | `state.sqlite3` | API reads |
| `api/healthz.py` | `state.sqlite3` | Health checks |
| `db/init_db.py` | `state.sqlite3` | Schema migrations |

### Migration Strategy

1. Abstract all `sqlite3.connect()` behind a `get_connection()` factory
2. Add PostgreSQL driver (`psycopg2` or `asyncpg`) as optional dependency
3. Use env var `DATABASE_URL` (default: `sqlite:///...` for backward compat)
4. Run both DBs in parallel during migration period

---

## 2. Redis Feature Cache

### Architecture

```
Nightly DAG (18:00 ET)
  ├─ Chronos-2 priors → Redis SET "priors:{date}" {json}
  ├─ VRP vol surface  → Redis SET "vol_surface:{date}" {json}
  └─ COOC T-1 features → Redis SET "cooc_features:{date}" {binary}

Morning inference (07:00, 09:20 ET)
  └─ Redis GET → instant O(1) retrieval (no GPU, no disk I/O)
```

### Implementation

- `pip install redis`
- New module: `backend/app/ml_platform/cache/redis_store.py`
- TTL: 48 hours (auto-expire stale priors)
- Fallback: if Redis unavailable, re-compute (current behavior)

---

## 3. Linux / Docker IBC Deployment

### Architecture

```
Docker Compose:
  ├─ ibgateway:   IBC + IB Gateway (headless, port 4002)
  ├─ algae-api:   FastAPI backend
  ├─ algae-orch:  Orchestrator daemon
  ├─ postgres:    PostgreSQL 16
  └─ redis:       Redis 7
```

### Benefits

- Eliminates TWS GUI memory leaks and KVM switch failures
- Enables proper process isolation and resource limits
- `docker compose up` replaces all `.bat` / `.ps1` scripts
- Health checks via Docker native restart policies
