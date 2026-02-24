# start_trading.ps1 — Launch backend API + orchestrator daemon for paper trading
# Usage: powershell -ExecutionPolicy Bypass -File scripts\start_trading.ps1
#
# Starts two processes:
#   1. Backend API server (uvicorn on port 8000)
#   2. Orchestrator daemon (polls every 60s, paper broker, telemetry enabled)
#
# Both write logs to backend/logs/. Press Ctrl+C to stop.

param(
    [string]$Broker = "stub",            # "stub" for paper, "ibkr" for live
    [int]$PollInterval = 60,             # seconds between orchestrator ticks
    [switch]$DryRun                      # pass -DryRun for noop mode
)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ROOT

# Set IBKR Gateway connection for paper trading (port 4002)
# TWS uses 7497 (live) / 7496 (paper); Gateway uses 4001 (live) / 4002 (paper)
if ($Broker -eq "ibkr" -and -not $env:IBKR_GATEWAY_URL) {
    $env:IBKR_GATEWAY_URL = "127.0.0.1:4002"
    Write-Host "  Set IBKR_GATEWAY_URL=$env:IBKR_GATEWAY_URL (Gateway paper)" -ForegroundColor Gray
}

# Ensure log directory exists
$logDir = Join-Path $ROOT "backend\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

$timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " ALGAIE Trading Stack Startup"          -ForegroundColor Cyan
Write-Host " Broker:   $Broker"                      -ForegroundColor Cyan
Write-Host " Poll:     ${PollInterval}s"             -ForegroundColor Cyan
Write-Host " Dry-run:  $DryRun"                      -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# --- 1. Start IBKR Gateway via IBC (if applicable) ---
if ($Broker -eq "ibkr") {
    $ibcPath = "C:\IBC\StartGateway.bat"
    if (Test-Path $ibcPath) {
        Write-Host "`n[0/2] Starting IBKR Gateway via IBC..." -ForegroundColor Green
        Start-Process -FilePath $ibcPath -WindowStyle Minimized
        Write-Host "  Waiting 20 seconds for Gateway to initialize and login..." -ForegroundColor Yellow
        Start-Sleep -Seconds 20
    }
    else {
        Write-Host "`n[0/2] Warning: C:\IBC\StartGateway.bat not found. Assuming Gateway is already running." -ForegroundColor Yellow
    }
}

# --- 2. Start Backend API Server ---
$apiLog = Join-Path $logDir "api_${timestamp}.log"
Write-Host "`n[1/2] Starting backend API on port 8000..." -ForegroundColor Green
$apiProc = Start-Process -FilePath "py" `
    -ArgumentList "-m", "uvicorn", "backend.app.api.main:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "info" `
    -WorkingDirectory $ROOT `
    -RedirectStandardOutput $apiLog `
    -RedirectStandardError (Join-Path $logDir "api_${timestamp}_err.log") `
    -PassThru -NoNewWindow

Write-Host "  API PID: $($apiProc.Id) | Log: $apiLog"

# Give uvicorn a moment to bind the port
Start-Sleep -Seconds 3

# Health check
try {
    $health = Invoke-RestMethod -Uri "http://localhost:8000/healthz" -TimeoutSec 5
    Write-Host "  Health: OK (elapsed $($health.elapsed_ms)ms)" -ForegroundColor Green
}
catch {
    Write-Host "  Warning: healthz not responding yet (may still be starting)" -ForegroundColor Yellow
}

# --- 2. Start Orchestrator Daemon ---
$orchLog = Join-Path $logDir "orchestrator_${timestamp}.log"
$orchArgs = @("-m", "backend.scripts.orchestrate", "--daemon", "--broker", $Broker, "--telemetry", "--poll-interval", $PollInterval)
if ($DryRun) { $orchArgs += "--dry-run" }

Write-Host "`n[2/2] Starting orchestrator daemon (broker=$Broker, poll=${PollInterval}s)..." -ForegroundColor Green
$orchProc = Start-Process -FilePath "py" `
    -ArgumentList $orchArgs `
    -WorkingDirectory $ROOT `
    -RedirectStandardOutput $orchLog `
    -RedirectStandardError (Join-Path $logDir "orchestrator_${timestamp}_err.log") `
    -PassThru -NoNewWindow

Write-Host "  Orchestrator PID: $($orchProc.Id) | Log: $orchLog"

Write-Host "`n========================================" -ForegroundColor Green
Write-Host " Both processes started!"                   -ForegroundColor Green
Write-Host " API:          http://localhost:8000"        -ForegroundColor Green
Write-Host " Frontend:     http://localhost:3000"        -ForegroundColor Green
Write-Host " Logs:         $logDir"                      -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host "`nTo stop both:"
Write-Host "  Stop-Process -Id $($apiProc.Id), $($orchProc.Id) -Force"
Write-Host "`nTo tail orchestrator log:"
Write-Host "  Get-Content '$orchLog' -Wait -Tail 20"

# Wait for either process to exit
Write-Host "`nMonitoring... (press Ctrl+C to stop)" -ForegroundColor Yellow
try {
    while (-not $apiProc.HasExited -and -not $orchProc.HasExited) {
        Start-Sleep -Seconds 5
    }
    if ($apiProc.HasExited) { Write-Host "API server exited with code $($apiProc.ExitCode)" -ForegroundColor Red }
    if ($orchProc.HasExited) { Write-Host "Orchestrator exited with code $($orchProc.ExitCode)" -ForegroundColor Red }
}
finally {
    # Cleanup on Ctrl+C
    if (-not $apiProc.HasExited) { Stop-Process -Id $apiProc.Id -Force -ErrorAction SilentlyContinue }
    if (-not $orchProc.HasExited) { Stop-Process -Id $orchProc.Id -Force -ErrorAction SilentlyContinue }
    Write-Host "Both processes stopped." -ForegroundColor Yellow
}
