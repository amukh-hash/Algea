#!/bin/bash
set -e

echo "=== Smoke Phase 2 ==="

# 1. Ingest Mock IV
echo "Ingesting IV..."
python backend/scripts/options/ingest_iv.py --ticker AAPL --start_date 2023-01-01 --end_date 2023-02-01 --dte 30

# 2. Build Gate Dataset
echo "Building Gate Dataset..."
python backend/scripts/options/build_gate_dataset.py

# 3. Tune Gate
echo "Tuning Gate..."
python backend/scripts/options/tune_options_gate.py

# 4. Run Hybrid Monitor (Mock Mode)
echo "Running Hybrid Monitor (Mock Mode)..."

export EQUITIES_MODE="mock"
# Generate mock execution
python backend/scripts/run_alga3_swing_hybrid_monitor.py --dry-run --ticker AAPL

echo "=== Phase 2 Smoke Complete ==="
