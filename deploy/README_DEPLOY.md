# Algaie Day-2 Operations: 30-Day Burn-In Deployment Guide

## 1. Host Timezone Verification

The host machine **must** run in UTC. Verify and set:

```bash
timedatectl set-timezone Etc/UTC
timedatectl status   # Confirm "Time zone: Etc/UTC"
```

## 2. Systemd Service Installation

```bash
sudo cp deploy/algaie_inference.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable algaie_inference
sudo systemctl start algaie_inference

# Verify TZ=UTC is inherited:
systemctl show algaie_inference -p Environment
```

**Key constraints:**
- `--workers 1` → prevents zombie CUDA contexts from POSIX forks
- `--limit-max-requests 10000` → forces Uvicorn to recycle the worker, releasing fragmented GPU memory
- `MemoryMax=32G` → kernel OOM-kills the process before it can destabilize the host

## 3. Cron Schedule Activation

```bash
sudo mkdir -p /var/log/algaie
crontab deploy/crontab.conf    # Or append to existing crontab
crontab -l                      # Verify entries
```

| Time (UTC) | Script | Purpose |
|------------|--------|---------|
| 17:00 M–F | `08_t1_resolution.py` | Resolve T+1 outcomes → write `ece_tracking` |
| 23:00 M–F | `07_nightly_run.py` | DAG orchestrator → reads ECE/MMD FSM circuit breakers |

The 6-hour gap eliminates SQLite WAL race conditions.

## 4. DST Audit Checklist — March 8, 2026

US clocks spring forward: EST (UTC-5) → EDT (UTC-4).

**Code is already DST-safe:**
- `_enforce_dte_flattening()` uses `ZoneInfo("US/Eastern")` → auto-adjusts
- `_is_auction_window()` uses `ZoneInfo("US/Eastern")` → auto-adjusts
- All `datetime.now()` calls use `timezone.utc`

**Operator verification on March 9:**
1. Confirm systemd service still has `TZ=UTC` active
2. Verify cron jobs fired at correct UTC times (check `/var/log/algaie/*.log` timestamps)
3. Confirm US market close shifted from 21:00 UTC → 20:00 UTC in orchestrator logs

## 5. Daily Telemetry Queries

### ECE Out-of-Sample Tracking

Run after 17:00 UTC (post-T+1 resolution):

```sql
SELECT
    sleeve,
    confidence_bin,
    COUNT(actual_outcome) as sample_size,
    AVG(predicted_probability) as stated_confidence,
    (CAST(SUM(actual_outcome) AS FLOAT) / COUNT(*)) as empirical_hit_rate,
    ABS(AVG(predicted_probability) - (CAST(SUM(actual_outcome) AS FLOAT) / COUNT(*))) as current_ece
FROM ece_tracking
WHERE actual_outcome IS NOT NULL
GROUP BY sleeve, confidence_bin;
```

**Halt condition:** `current_ece > 0.10` in the `0.80-0.90` bin triggers `HALTED_ECE_BREACH`.

### MMD Concept Drift

```bash
grep "HALTED_DRIFT" /var/log/algaie/nightly_run.log
```

If triggered during low-volatility sessions, the RBF kernel bandwidth (σ) in `liveguard_baselines.py` is too narrow.

### VRAM Plateau

```bash
nvidia-smi dmon -s m   # Monitor cuda:1 (RTX 4070 Super)
```

VRAM must stay horizontally asymptotic (e.g., 4.1–4.3 GB). If it stair-steps upward by ≥50 MB/day, a tensor leak exists in `server.py` (missing `.detach().cpu()`). This triggers `HALTED_OOM`.

## 6. Broker Disconnect Resilience

The `IBKRBrokerAdapter` now includes `tenacity` exponential backoff retry logic on all network-bound methods. IBKR gateway resets (~23:45 EST daily) are handled automatically with up to 10 retries over ~10 minutes.

## 7. Log Rotation

Logs rotate at midnight UTC with 7-day retention via `TimedRotatingFileHandler`. No manual cleanup needed. Log directory: `backend/logs/` (local) or `LOG_DIR` env var (production).
