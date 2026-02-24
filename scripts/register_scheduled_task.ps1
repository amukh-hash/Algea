# register_scheduled_task.ps1 — Register a Windows Scheduled Task for Algaie trading
# Must be run as Administrator (elevated prompt).
#
# Usage: powershell -ExecutionPolicy Bypass -File scripts\register_scheduled_task.ps1
#
# Creates a task that runs at 6:30 AM ET on weekdays to start the full trading stack
# before PREMARKET (7:00 AM). The task runs under the current user account.

param(
    [string]$TaskName = "Algaie-TradingStack",
    [string]$StartTime = "06:30",          # 6:30 AM local time (before 7:00 PREMARKET)
    [string]$Broker = "stub",              # "stub" for paper, "ibkr" for live
    [switch]$Remove                         # pass -Remove to unregister the task
)

$ROOT = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$scriptPath = Join-Path $ROOT "scripts\autostart_trading.ps1"

if ($Remove) {
    Write-Host "Removing scheduled task '$TaskName'..." -ForegroundColor Yellow
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue
    Write-Host "Done." -ForegroundColor Green
    return
}

# Verify script exists
if (-not (Test-Path $scriptPath)) {
    Write-Error "autostart_trading.ps1 not found at: $scriptPath"
    return
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host " Registering Algaie Scheduled Task"      -ForegroundColor Cyan
Write-Host " Task:     $TaskName"                      -ForegroundColor Cyan
Write-Host " Time:     $StartTime (weekdays)"          -ForegroundColor Cyan
Write-Host " Broker:   $Broker"                        -ForegroundColor Cyan
Write-Host " Script:   $scriptPath"                    -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Build the action — run PowerShell with the autostart script
$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument "-ExecutionPolicy Bypass -WindowStyle Hidden -File `"$scriptPath`" -Broker $Broker" `
    -WorkingDirectory $ROOT

# Trigger: weekdays at the specified time
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday -At $StartTime

# Settings: don't stop if on battery, allow wake, kill after 14 hours (market day)
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -WakeToRun `
    -ExecutionTimeLimit (New-TimeSpan -Hours 14) `
    -RestartCount 2 `
    -RestartInterval (New-TimeSpan -Minutes 5)

# Register under current user (no password prompt for interactive logon)
$principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

try {
    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Settings $settings `
        -Principal $principal `
        -Description "Launches Algaie trading stack (API + Orchestrator + Frontend) before market open." `
        -Force

    Write-Host "`n✓ Task '$TaskName' registered successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Schedule: Every weekday at $StartTime" -ForegroundColor White
    Write-Host "Verify:   Get-ScheduledTaskInfo -TaskName '$TaskName'" -ForegroundColor Gray
    Write-Host "Test:     Start-ScheduledTask -TaskName '$TaskName'" -ForegroundColor Gray
    Write-Host "Remove:   .\scripts\register_scheduled_task.ps1 -Remove" -ForegroundColor Gray
    Write-Host ""
    Write-Host "The stack will auto-start 30 minutes before PREMARKET each trading day." -ForegroundColor Yellow
}
catch {
    Write-Error "Failed to register task. Ensure you are running as Administrator."
    Write-Error $_.Exception.Message
}
