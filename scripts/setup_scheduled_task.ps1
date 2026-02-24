param(
    [switch]$Unregister     # Remove the task instead of creating it
)

# setup_scheduled_task.ps1 - Register a Windows Task Scheduler entry
# to auto-start the orchestrator daemon at 6:50 AM ET on weekdays.
#
# MUST be run as Administrator (elevated PowerShell).
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_scheduled_task.ps1 [-Unregister]

$ErrorActionPreference = "Stop"
$TaskName = "ALGAIE-TradingOrchestrator"

if ($Unregister) {
    Write-Host "Removing scheduled task '$TaskName'..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Done." -ForegroundColor Green
    exit 0
}

$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

# The action: run the start_trading.ps1 script
$scriptPath = Join-Path $ROOT "scripts\start_trading.ps1"
if (-not (Test-Path $scriptPath)) {
    Write-Error "Cannot find $scriptPath - run this from the repo root."
    exit 1
}

$Action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`" -Broker ibkr" `
    -WorkingDirectory $ROOT

# Trigger: weekdays at 6:50 AM (local time - set your system to Eastern)
$Trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At "06:50"

# Settings: allow on battery, don't stop on idle, restart on failure
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Hours 14)

# Register
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Start ALGAIE backend API + orchestrator daemon for paper trading at 6:50 AM on weekdays." `
    -Force

Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host " Scheduled Task Created" -ForegroundColor Green
Write-Host " Name:    $TaskName" -ForegroundColor Green
Write-Host " Trigger: Weekdays at 6:50 AM" -ForegroundColor Green
Write-Host " Script:  $scriptPath" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "To verify:  Get-ScheduledTask -TaskName '$TaskName' | Format-List"
Write-Host "To run now: Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "To remove:  powershell -File $($MyInvocation.MyCommand.Path) -Unregister"
