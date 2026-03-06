# Algaie Trading System - Unified Launcher
# Launches: IB Gateway + Backend API + Native Frontend
$ErrorActionPreference = "Continue"

$ROOT = "C:\Users\crick\Documents\Workshop\Algaie"
$VENV_PYTHON = "$ROOT\venv_gpu\Scripts\python.exe"
$BACKEND_MODULE = "backend.app.api.main:app"
$FRONTEND_EXE = "$ROOT\native_frontend\build\Release\Algaie_Sim.exe"
$IB_GATEWAY_EXE = "C:\Jts\ibgateway\1043\ibgateway.exe"
$IB_GATEWAY_ARGS = '-J-DjtsConfigDir="C:\Jts\ibgateway\1043"'

Write-Host ""
Write-Host "  ================================================" -ForegroundColor Cyan
Write-Host "       ALGAIE TRADING SYSTEM LAUNCHER" -ForegroundColor Cyan
Write-Host "  ================================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Launch IB Gateway
Write-Host "[1/3] Launching IB Gateway..." -ForegroundColor Yellow

$ibRunning = Get-Process -Name "ibgateway" -ErrorAction SilentlyContinue
if ($ibRunning) {
    Write-Host "  > IB Gateway already running (PID $($ibRunning.Id))" -ForegroundColor Green
}
else {
    if (Test-Path $IB_GATEWAY_EXE) {
        Start-Process -FilePath $IB_GATEWAY_EXE -ArgumentList $IB_GATEWAY_ARGS -WorkingDirectory "C:\Jts\ibgateway\1043"
        Write-Host "  > IB Gateway started. Please log in if prompted." -ForegroundColor Green
        Write-Host "  > Waiting 10 seconds for IB Gateway to initialize..." -ForegroundColor Gray
        Start-Sleep 10
    }
    else {
        Write-Host "  > WARNING: IB Gateway not found at $IB_GATEWAY_EXE" -ForegroundColor Red
        Write-Host "  > Continuing without IB Gateway..." -ForegroundColor Red
    }
}

# Step 2: Start Backend
Write-Host "[2/3] Starting backend API server..." -ForegroundColor Yellow

$backendRunning = $false
try {
    $health = Invoke-RestMethod -Uri "http://127.0.0.1:8000/healthz" -TimeoutSec 2 -ErrorAction Stop
    Write-Host "  > Backend already running" -ForegroundColor Green
    $backendRunning = $true
}
catch {
    $env:PYTHONPATH = $ROOT
    $backendProc = Start-Process -FilePath $VENV_PYTHON `
        -ArgumentList "-m", "uvicorn", $BACKEND_MODULE, "--host", "0.0.0.0", "--port", "8000" `
        -WorkingDirectory $ROOT `
        -WindowStyle Minimized `
        -PassThru

    Write-Host "  > Backend starting (PID $($backendProc.Id))..." -ForegroundColor Green

    for ($i = 0; $i -lt 15; $i++) {
        Start-Sleep 1
        try {
            Invoke-RestMethod -Uri "http://127.0.0.1:8000/healthz" -TimeoutSec 2 -ErrorAction Stop | Out-Null
            $backendRunning = $true
            break
        }
        catch { }
    }

    if ($backendRunning) {
        Write-Host "  > Backend ready!" -ForegroundColor Green
    }
    else {
        Write-Host "  > WARNING: Backend may not be ready yet" -ForegroundColor Red
    }
}

# Step 3: Launch Native Frontend
Write-Host "[3/3] Launching Algaie native frontend..." -ForegroundColor Yellow

$env:QML_IMPORT_PATH = "$ROOT\native_frontend\build\Release\qml"

$feRunning = Get-Process -Name "Algaie_Sim" -ErrorAction SilentlyContinue
if ($feRunning) {
    Write-Host "  > Frontend already running (PID $($feRunning.Id))" -ForegroundColor Green
}
else {
    if (Test-Path $FRONTEND_EXE) {
        $feProc = Start-Process -FilePath $FRONTEND_EXE -PassThru
        Write-Host "  > Frontend started (PID $($feProc.Id))" -ForegroundColor Green
    }
    else {
        Write-Host "  > ERROR: Frontend not found at $FRONTEND_EXE" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "  ================================================" -ForegroundColor Green
Write-Host "       ALL SYSTEMS LAUNCHED" -ForegroundColor Green
Write-Host "  ================================================" -ForegroundColor Green
Write-Host "  IB Gateway  > port 4002 (paper)" -ForegroundColor Green
Write-Host "  Backend API > http://127.0.0.1:8000" -ForegroundColor Green
Write-Host "  Frontend    > D3D11 native" -ForegroundColor Green
Write-Host "  Auto-connect fires in ~3 seconds" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to close this window"
