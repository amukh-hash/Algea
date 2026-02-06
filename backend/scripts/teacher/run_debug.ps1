$env:PYTHONPATH = $PWD.Path
$env:GOLD_MAX_FILES="10"
$env:GOLD_MAX_STEPS="50"
$env:GOLD_WARMUP="10"
$env:GOLD_CHECKPOINT_EVERY="50"
$env:GOLD_LOG_EVERY="10"
$env:USE_QLORA="0"
$env:GOLD_BATCH_SIZE="2"
$env:GOLD_GRAD_ACCUM="1"
$env:GOLD_CONTEXT="128"
$env:GOLD_LR="1e-4"
$env:TEACHER_E_GOLD_OUTDIR="backend/models/teacher_e/debug_run"

Write-Host "Starting Debug Run..."
python backend/scripts/teacher/phase1_train_teacher_gold.py
