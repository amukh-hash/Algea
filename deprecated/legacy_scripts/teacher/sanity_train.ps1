$env:GOLD_MAX_FILES="1"
$env:GOLD_MAX_STEPS="200"
$env:GOLD_LOG_EVERY="10"
$env:GOLD_BATCH_SIZE="4"
$env:GOLD_GRAD_ACCUM="1"
$env:GOLD_VAL_QUICK_EVERY="50"
$env:TEACHER_E_GOLD_OUTDIR="backend/models/teacher_e/sanity_check"
.\backend\scripts\teacher\run_overnight.ps1
