# teardown_stack.ps1 — Daily post-market teardown of the Algae trading stack
# Registered as: Algae_EOD_Teardown (4:15 PM M-F)
#
# Forcefully terminates TWS, Backend API, and any Python processes
# so that the 06:25 AM Algae_TWS_Boot trigger boots from a clean slate.

$ErrorActionPreference = "SilentlyContinue"

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Initiating Daily Teardown..." -ForegroundColor Yellow

# 1. Kill the IB TWS (Java/javaw)
$javaProcs = Get-Process -Name "java", "javaw" -ErrorAction SilentlyContinue
if ($javaProcs) {
    $javaProcs | Stop-Process -Force
    Write-Host "  IB TWS killed (PIDs: $($javaProcs.Id -join ', '))"
}
else {
    Write-Host "  IB TWS: not running"
}

# 2. Kill Python processes spawned by the trading stack
$pythonProcs = Get-Process -Name "python", "py" -ErrorAction SilentlyContinue
if ($pythonProcs) {
    $pythonProcs | Stop-Process -Force
    Write-Host "  Python processes killed (PIDs: $($pythonProcs.Id -join ', '))"
}
else {
    Write-Host "  Python: no processes running"
}

# 3. Brief pause to let ports release
Start-Sleep -Seconds 2

# 4. Verify ports are free
$port8000 = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
$port7497 = Get-NetTCPConnection -LocalPort 7497 -ErrorAction SilentlyContinue

if (-not $port8000 -and -not $port7497) {
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Teardown complete. Ports 8000 and 7497 released for tomorrow." -ForegroundColor Green
}
else {
    if ($port8000) { Write-Host "  WARNING: Port 8000 still held by PID $($port8000.OwningProcess)" -ForegroundColor Red }
    if ($port7497) { Write-Host "  WARNING: Port 7497 still held by PID $($port7497.OwningProcess)" -ForegroundColor Red }
}

# 5. Log rotation — purge logs older than 7 days
$logDir = Join-Path $PSScriptRoot "..\backend\logs"
if (Test-Path $logDir) {
    $stale = Get-ChildItem $logDir -File | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) }
    if ($stale) {
        $stale | Remove-Item -Force
        Write-Host "  Log rotation: purged $($stale.Count) files older than 7 days"
    }
}

# 6. Purge IBC logs older than 7 days
$ibcLogDir = "C:\IBC\Logs"
if (Test-Path $ibcLogDir) {
    $staleIbc = Get-ChildItem $ibcLogDir -File | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-7) }
    if ($staleIbc) {
        $staleIbc | Remove-Item -Force
        Write-Host "  IBC log rotation: purged $($staleIbc.Count) files older than 7 days"
    }
}

Write-Host "[$(Get-Date -Format 'HH:mm:ss')] Daily teardown finished." -ForegroundColor Green
