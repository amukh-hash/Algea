# ─────────────────────────────────────────────────────────────────────
# Launch_Algae.ps1 — Production Deployment Bootstrapper
#
# Dynamically sets the QSG_SHADER_CACHE_PATH environment variable
# based on the executable's ProductVersion to prevent Vulkan pipeline
# cache corruption after GPU driver updates.
#
# Deploy via: Windows GPO, Intune, or desktop shortcut Target field.
# ─────────────────────────────────────────────────────────────────────
param(
    [string]$Target = "Live",                    # "Live" or "Sim"
    [string]$InstallDir = "$PSScriptRoot"         # Directory containing Algae_*.exe
)

$ErrorActionPreference = "Stop"

# ── Resolve executable path ──────────────────────────────────────────
$exeName = if ($Target -eq "Sim") { "Algae_Sim.exe" } else { "Algae_Live.exe" }
$exePath = Join-Path $InstallDir $exeName

if (-not (Test-Path $exePath)) {
    Write-Error "Executable not found: $exePath"
    exit 1
}

# ── Versioned Vulkan shader cache ────────────────────────────────────
# Qt RHI compiles SPIR-V shaders to disk cache on first launch.
# If the GPU driver updates silently, cached pipeline layouts become
# invalid and cause EXCEPTION_ACCESS_VIOLATION (0xC0000005) on startup.
# Isolating by version forces a clean cache rebuild on each update.
$versionInfo = [System.Diagnostics.FileVersionInfo]::GetVersionInfo($exePath)
$appVersion = $versionInfo.ProductVersion
if (-not $appVersion) { $appVersion = "unknown" }

$cachePath = Join-Path $env:LOCALAPPDATA "AlgaeTrading\ShaderCache_v$appVersion"
if (-not (Test-Path $cachePath)) {
    New-Item -ItemType Directory -Force -Path $cachePath | Out-Null
    Write-Host "[Algae] Created shader cache: $cachePath"
}

# Inject environment variable strictly into this process context
[System.Environment]::SetEnvironmentVariable("QSG_SHADER_CACHE_PATH", $cachePath, "Process")

# ── Crash dump directory ─────────────────────────────────────────────
$crashDir = Join-Path $InstallDir "crash_dumps"
if (-not (Test-Path $crashDir)) {
    New-Item -ItemType Directory -Force -Path $crashDir | Out-Null
}

# ── Launch ───────────────────────────────────────────────────────────
Write-Host "[Algae] Launching $exeName (v$appVersion)"
Write-Host "[Algae] Shader cache: $cachePath"
Write-Host "[Algae] Crash dumps: $crashDir"

Start-Process -FilePath $exePath -WorkingDirectory $InstallDir -NoNewWindow -Wait
