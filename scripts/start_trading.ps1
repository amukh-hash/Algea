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

# --- DISASTER RECOVERY TIME GUARD ---
# Prevents the stack from booting on weekends or outside trading hours.
# Essential when using the "At Log On" Task Scheduler trigger for DR.
$now = Get-Date
$startWindow = (Get-Date).Date.AddHours(6).AddMinutes(25)  # 06:25 AM
$endWindow = (Get-Date).Date.AddHours(16).AddMinutes(10) # 04:10 PM
$isWeekday = ($now.DayOfWeek -ge [System.DayOfWeek]::Monday) -and ($now.DayOfWeek -le [System.DayOfWeek]::Friday)

if (-not $isWeekday -or ($now -lt $startWindow) -or ($now -gt $endWindow)) {
    Write-Host "System booted outside active trading window ($now). Aborting auto-start." -ForegroundColor Yellow
    exit 0
}
# ------------------------------------

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ROOT

# Set IBKR TWS connection for paper trading (port 7497)
# TWS uses 7497 (live) / 7496 (paper); Gateway uses 4001 (live) / 4002 (paper)
if ($Broker -eq "ibkr" -and -not $env:IBKR_GATEWAY_URL) {
    $env:IBKR_GATEWAY_URL = "127.0.0.1:7497"
    Write-Host "  Set IBKR_GATEWAY_URL=$env:IBKR_GATEWAY_URL (TWS paper)" -ForegroundColor Gray
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

# --- 1. Start IBKR TWS via IBC (if applicable) ---
if ($Broker -eq "ibkr") {
    $ibcPath = "C:\IBC\StartTWS.bat"
    if (Test-Path $ibcPath) {
        Write-Host "`n[0/2] Starting IBKR TWS via IBC..." -ForegroundColor Green
        Start-Process -FilePath $ibcPath -WindowStyle Minimized
        Write-Host "  Waiting 30 seconds for TWS to initialize and login..." -ForegroundColor Yellow
        Start-Sleep -Seconds 30
    }
    else {
        Write-Host "`n[0/2] Warning: C:\IBC\StartTWS.bat not found. Assuming TWS is already running." -ForegroundColor Yellow
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

# Wait for orchestrator daemon exit or Ctrl+C, while watchdog-restarting the API
Write-Host "`nMonitoring... (press Ctrl+C to stop)" -ForegroundColor Yellow
Write-Host "  API watchdog: auto-restart on crash (5s cooldown)" -ForegroundColor Gray

function Start-ApiProcess {
    $ts = Get-Date -Format "yyyy-MM-dd_HHmmss"
    $log = Join-Path $logDir "api_${ts}.log"
    $err = Join-Path $logDir "api_${ts}_err.log"
    $proc = Start-Process -FilePath "py" `
        -ArgumentList "-m", "uvicorn", "backend.app.api.main:app", "--host", "127.0.0.1", "--port", "8000", "--log-level", "info" `
        -WorkingDirectory $ROOT `
        -RedirectStandardOutput $log `
        -RedirectStandardError $err `
        -PassThru -NoNewWindow
    Write-Host "  API (re)started PID=$($proc.Id) | Log: $log" -ForegroundColor Green
    return $proc
}

try {
    while ($true) {
        if ($apiProc.HasExited) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] API process died (exit code $($apiProc.ExitCode)) — restarting in 5s..." -ForegroundColor Red
            Start-Sleep -Seconds 5
            $apiProc = Start-ApiProcess
        }
        if ($orchProc.HasExited) {
            Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Orchestrator daemon exited with code $($orchProc.ExitCode)" -ForegroundColor Red
            break
        }
        Start-Sleep -Seconds 5
    }
}
finally {
    # Cleanup on Ctrl+C
    if (-not $apiProc.HasExited) { Stop-Process -Id $apiProc.Id -Force -ErrorAction SilentlyContinue }
    if (-not $orchProc.HasExited) { Stop-Process -Id $orchProc.Id -Force -ErrorAction SilentlyContinue }
    Write-Host "Both processes stopped." -ForegroundColor Yellow
}
