#!/bin/bash
set -e

echo "=== Phase 1 Smoke Test ==="

export PYTHONPATH=$PYTHONPATH:.

# 0. Generate Dummy Data
echo "[0] Generating Dummy Data..."
python backend/scripts/generate_dummy_data.py

# 1. Build Breadth Context
echo "[1] Building Breadth Context..."
python backend/scripts/build_breadth_context.py

# 2. Build MarketFrames
echo "[2] Building MarketFrames..."
# Only build AAPL to save time, or all dummy ones
python backend/scripts/build_marketframe.py --ticker AAPL

# 3. Fit Preproc
echo "[3] Fitting Preproc..."
python backend/scripts/fit_preproc.py --start_date 2023-01-01 --end_date 2023-01-15 --data_dir backend/data/marketframe

# 4. Train Teacher Baseline
echo "[4] Training Teacher Baseline..."
python backend/scripts/train_teacher_e.py --data_dir backend/data/marketframe --preproc_path backend/models/preproc/preproc_v1.json --output_path backend/models/teacher_e/teacher_smoke.pt

# 5. Train Student Baseline
echo "[5] Training Student Baseline..."
python backend/scripts/train_student_baseline.py --data_dir backend/data/marketframe --preproc_path backend/models/preproc/preproc_v1.json --output_path backend/models/student/student_smoke.pt

# 6. Eval Student
echo "[6] Evaluating Student..."
python backend/scripts/eval_student_rolling.py --model_path backend/models/student/student_smoke.pt --preproc_path backend/models/preproc/preproc_v1.json --data_dir backend/data/marketframe

# 7. Run Engine (Paper Trading)
echo "[7] Running Paper Trading Engine..."
python backend/scripts/run_alga3_swing_equities.py --ticker AAPL --data_dir backend/data/marketframe --model_path backend/models/student/student_smoke.pt --preproc_path backend/models/preproc/preproc_v1.json

# 8. Validate Artifacts
echo "[8] Validating Artifacts..."
python backend/scripts/validate_phase1_artifacts.py

echo "=== Smoke Test Complete ==="
