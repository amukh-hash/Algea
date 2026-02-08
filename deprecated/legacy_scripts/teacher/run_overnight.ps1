# Overnight Training Pipeline
# Single Continuous Run (10k steps)

$env:PYTHONPATH = $PWD.Path
$env:GOLD_MAX_FILES="1000"
$env:GOLD_MAX_STEPS="10000"     # 10k steps
$env:GOLD_VAL_QUICK_EVERY="500" # Validation every 500 steps
$env:GOLD_SAVE_LAST_EVERY="500"
$env:GOLD_LOG_EVERY="100"       # Logging every 100 steps
$env:USE_QLORA="0"
$env:GOLD_BATCH_SIZE="8"   
$env:GOLD_GRAD_ACCUM="4"   # 32 Effective Batch
$env:GOLD_CONTEXT="1024"
$env:GOLD_PRED="10"        
$env:GOLD_LR="5e-5"
$env:GOLD_FILE_CACHE="1000"

Write-Host "=========================================="
Write-Host "STARTING TRAINING (Gold Phase)"
Write-Host "Output: backend/models/teacher_e/gold_full"
Write-Host "=========================================="

$env:TEACHER_E_GOLD_OUTDIR="backend/models/teacher_e/gold_full"
python backend/scripts/teacher/phase1_train_teacher_gold.py

if ($LASTEXITCODE -ne 0) {
    Write-Host "Training Failed! Exiting."
    exit $LASTEXITCODE
}

Write-Host "=========================================="
Write-Host "TRAINING COMPLETE."
Write-Host "=========================================="
