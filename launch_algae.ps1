# -----------------------------------------------------------------
# Algae Trading System - One-Click Launcher
# -----------------------------------------------------------------

$ErrorActionPreference = "Continue"
$ProjectRoot = "C:\Users\crick\ResolveLabs\Algae"
$BackendBat = Join-Path $ProjectRoot "start_backend.bat"
$FrontendExe = Join-Path $ProjectRoot "native_frontend\build\Release\Algae_Sim.exe"

Write-Host ""
Write-Host "  ===============================" -ForegroundColor Cyan
Write-Host "   ALGAE TRADING SYSTEM LAUNCHER  " -ForegroundColor Cyan
Write-Host "  ===============================" -ForegroundColor Cyan
Write-Host ""

# -- 1. Start Backend -------------------------------------------------
Write-Host "[1/3] Starting backend server..." -ForegroundColor Yellow

$backendAlive = $false
try {
    $null = Invoke-RestMethod -Uri "http://127.0.0.1:8000/healthz" -TimeoutSec 2
    $backendAlive = $true
    Write-Host "  Backend already running - skipping" -ForegroundColor Green
}
catch {}

if (-not $backendAlive) {
    Start-Process $BackendBat -WindowStyle Minimized
    Write-Host "  Backend process launched" -ForegroundColor Green

    Write-Host "  Waiting for healthcheck" -ForegroundColor Gray -NoNewline
    $ready = $false
    for ($i = 0; $i -lt 30; $i++) {
        Start-Sleep -Seconds 1
        Write-Host "." -NoNewline -ForegroundColor Gray
        try {
            $null = Invoke-RestMethod -Uri "http://127.0.0.1:8000/healthz" -TimeoutSec 2
            $ready = $true
            break
        }
        catch {}
    }
    Write-Host ""
    if ($ready) {
        Write-Host "  Backend is healthy" -ForegroundColor Green
    }
    else {
        Write-Host "  [WARN] Backend healthcheck timed out" -ForegroundColor Red
    }
}

# -- 2. Connect to IB Gateway -----------------------------------------
Write-Host ""
Write-Host "[2/3] Connecting to IB Gateway..." -ForegroundColor Yellow

try {
    $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/control/broker/connect" -Method Post -ContentType "application/json" -Body "{}" -TimeoutSec 15

    if ($resp.connected -eq $true) {
        $msg = "  IBKR connected - " + $resp.status
        if ($resp.mode) { $msg += " | " + $resp.mode }
        if ($null -ne $resp.positions) { $msg += " | " + $resp.positions + " positions" }
        Write-Host $msg -ForegroundColor Green
    }
    else {
        Write-Host ("  IBKR not available: " + $resp.error) -ForegroundColor Red
        Write-Host "  (Make sure IB Gateway / TWS is running on port 7497)" -ForegroundColor Gray
    }
}
catch {
    Write-Host "  IBKR connection failed - is IB Gateway running?" -ForegroundColor Red
    Write-Host "  (The frontend will retry automatically on startup)" -ForegroundColor Gray
}

# -- 3. Launch Native Frontend ----------------------------------------
Write-Host ""
Write-Host "[3/3] Launching native frontend..." -ForegroundColor Yellow

$existing = Get-Process -Name "Algae_Sim" -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host ("  Algae_Sim already running (PID: " + $existing.Id + ")") -ForegroundColor Green
}
elseif (Test-Path $FrontendExe) {
    Start-Process -FilePath $FrontendExe -WorkingDirectory (Split-Path $FrontendExe)
    Start-Sleep -Seconds 3
    $fe = Get-Process -Name "Algae_Sim" -ErrorAction SilentlyContinue
    if ($fe) {
        Write-Host ("  Algae_Sim launched (PID: " + $fe.Id + ")") -ForegroundColor Green
    }
    else {
        Write-Host "  [WARN] Algae_Sim may have failed to start" -ForegroundColor Red
    }
}
else {
    Write-Host ("  [ERROR] Frontend not found: " + $FrontendExe) -ForegroundColor Red
}

# -- Done --------------------------------------------------------------
Write-Host ""
Write-Host "  ===============================" -ForegroundColor Green
Write-Host "   ALL SYSTEMS LAUNCHED            " -ForegroundColor Green
Write-Host "   Backend:  http://127.0.0.1:8000 " -ForegroundColor Green
Write-Host "   ZMQ:      tcp://127.0.0.1:5556  " -ForegroundColor Green
Write-Host "   Frontend: Algae [SIMULATION]   " -ForegroundColor Green
Write-Host "  ===============================" -ForegroundColor Green
Write-Host ""
Write-Host "  Press any key to close this window..." -ForegroundColor Gray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
