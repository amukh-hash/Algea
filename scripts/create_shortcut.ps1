# create_shortcut.ps1 — Create a Desktop shortcut for ALGAIE Dashboard
# Usage: powershell -ExecutionPolicy Bypass -File scripts\create_shortcut.ps1

$ErrorActionPreference = "Stop"
$WshShell = New-Object -ComObject WScript.Shell
$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutFile = Join-Path $DesktopPath "Algea Dashboard v4.0.lnk"
$TargetFile = Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "ALGAIE_DASHBOARD.bat"
$RepoRoot = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)

$Shortcut = $WshShell.CreateShortcut($ShortcutFile)
$Shortcut.TargetPath = $TargetFile
$Shortcut.WorkingDirectory = $RepoRoot
$Shortcut.WindowStyle = 7  # Minimized (so the launcher window hides away)
$Shortcut.Description = "Launch Algea Trading Stack v4.0 (Backend, Orchestrator, Frontend)"
$Shortcut.IconLocation = "shell32.dll,306"  # A generic 'graph/chart' looking icon from system
$Shortcut.Save()

Write-Host "Shortcut created at: $ShortcutFile" -ForegroundColor Green
Write-Host "You can now pin this to your start menu or taskbar by right-clicking it on the Desktop." -ForegroundColor Gray
