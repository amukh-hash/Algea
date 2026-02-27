# autostart_trading.ps1 — Headless launcher for Windows Task Scheduler
# Starts the full Algea trading stack (API + Orchestrator + Frontend dev server)
# Designed to run unattended before market open.
#
# Usage (manual): powershell -ExecutionPolicy Bypass -File scripts\autostart_trading.ps1
# Usage (task):   Registered via scripts\register_scheduled_task.ps1

param(
    [string]$Broker = "stub",       # "stub" for paper, "ibkr" for live
    [switch]$SkipFrontend           # skip npm run dev if frontend isn't needed
)

$ErrorActionPreference = "Continue"
$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $ROOT

# Set IBKR Gateway connection for paper trading (port 4002)
# TWS uses 7497 (live) / 7496 (paper); Gateway uses 4001 (live) / 4002 (paper)
if (-not $env:IBKR_GATEWAY_URL) {
    $env:IBKR_GATEWAY_URL = "127.0.0.1:4002"
}

$timestamp = Get-Date -Format "yyyy-MM-dd_HHmmss"
$logDir = Join-Path $ROOT "backend\logs"
if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }

$masterLog = Join-Path $logDir "autostart_${timestamp}.log"

function Log($msg) {
    $entry = "$(Get-Date -Format 'HH:mm:ss') $msg"
    Write-Host $entry
    Add-Content -Path $masterLog -Value $entry
}

Log "=== ALGAIE Auto-Start ==="
Log "Broker: $Broker | Root: $ROOT"

# ── Kill any stale processes on our ports ──
Log "Cleaning up stale processes on ports 8000 and 3000..."
foreach ($port in @(8000, 3000)) {
    Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique |
    ForEach-Object {
        Log "  Killing PID $_ on port $port"
        taskkill /F /PID $_ 2>$null
    }
}
Start-Sleep -Seconds 2

# ── 1. Start Backend API ──
$apiLog = Join-Path $logDir "api_${timestamp}.log"
$apiErrLog = Join-Path $logDir "api_${timestamp}_err.log"
Log "[1/3] Starting API server on port 8000..."
$apiProc = Start-Process -FilePath "python" `
    -ArgumentList "-m", "uvicorn", "backend.app.api.main:app", "--host", "127.0.0.1", "--port", "8000", "--workers", "4", "--log-level", "info" `
    -WorkingDirectory $ROOT `
    -RedirectStandardOutput $apiLog `
    -RedirectStandardError $apiErrLog `
    -PassThru -WindowStyle Hidden
Log "  API PID: $($apiProc.Id)"

Start-Sleep -Seconds 5

# Health check with retries
$healthy = $false
for ($i = 1; $i -le 5; $i++) {
    try {
        $h = Invoke-RestMethod -Uri "http://localhost:8000/healthz" -TimeoutSec 3
        Log "  Health: OK (attempt $i)"
        $healthy = $true
        break
    }
    catch {
        Log "  Health check attempt $i failed, retrying..."
        Start-Sleep -Seconds 2
    }
}
if (-not $healthy) { Log "  WARNING: API may not be healthy" }

# ── 2. Start Orchestrator Daemon ──
$orchLog = Join-Path $logDir "orchestrator_${timestamp}.log"
$orchErrLog = Join-Path $logDir "orchestrator_${timestamp}_err.log"
$orchArgs = @("-m", "backend.scripts.orchestrate", "--daemon", "--broker", $Broker, "--telemetry", "--poll-interval", "60")

Log "[2/3] Starting orchestrator daemon (broker=$Broker)..."
$orchProc = Start-Process -FilePath "python" `
    -ArgumentList $orchArgs `
    -WorkingDirectory $ROOT `
    -RedirectStandardOutput $orchLog `
    -RedirectStandardError $orchErrLog `
    -PassThru -WindowStyle Hidden
Log "  Orchestrator PID: $($orchProc.Id)"

# ── 3. Start Frontend (optional) ──
if (-not $SkipFrontend) {
    $feLog = Join-Path $logDir "frontend_${timestamp}.log"
    $feErrLog = Join-Path $logDir "frontend_${timestamp}_err.log"
    Log "[3/3] Starting frontend dev server on port 3000..."
    $feProc = Start-Process -FilePath "npm" `
        -ArgumentList "run", "dev" `
        -WorkingDirectory (Join-Path $ROOT "frontend") `
        -RedirectStandardOutput $feLog `
        -RedirectStandardError $feErrLog `
        -PassThru -WindowStyle Hidden
    Log "  Frontend PID: $($feProc.Id)"
}
else {
    Log "[3/3] Frontend skipped (-SkipFrontend)"
}

# ── Write PID file for easy cleanup ──
$pidFile = Join-Path $ROOT "backend\logs\running_pids.txt"
$pids = @($apiProc.Id, $orchProc.Id)
if ($feProc) { $pids += $feProc.Id }
$pids -join "`n" | Set-Content $pidFile
Log "PIDs written to $pidFile"

Log "=== Auto-Start Complete ==="
Log "API:          http://localhost:8000"
Log "Frontend:     http://localhost:3000"
Log "Dashboard:    http://localhost:3000/execution"
Log "Logs:         $logDir"
