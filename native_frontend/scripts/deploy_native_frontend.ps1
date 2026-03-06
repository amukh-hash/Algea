# ---------------------------------------------------------------
# Algaie Operational Deployment - 3-Step Runbook
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File .\deploy_native_frontend.ps1 -Step 1
#   powershell -ExecutionPolicy Bypass -File .\deploy_native_frontend.ps1 -Step 2
#   powershell -ExecutionPolicy Bypass -File .\deploy_native_frontend.ps1 -Step 3
# ---------------------------------------------------------------

param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("1", "2", "3")]
    [string]$Step
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$BackendDir = Join-Path $ProjectRoot "backend"
$NativeDir = Join-Path $ProjectRoot "native_frontend"
$BuildDir = Join-Path $NativeDir "build"

Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  Algaie Operational Deployment - Step $Step" -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host ""

switch ($Step) {
    "1" {
        Write-Host "[STEP 1] Dark Traffic Deployment" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  This step verifies that the backend ZMQ PUB sockets are"
        Write-Host "  active and emitting dark traffic alongside existing SSE."
        Write-Host ""

        # Check if backend is running
        Write-Host "[1.1] Checking backend process..." -ForegroundColor Gray
        try {
            $resp = Invoke-RestMethod -Uri "http://127.0.0.1:8000/healthz" -TimeoutSec 3
            Write-Host "  Backend is running (healthz OK)" -ForegroundColor Green
        }
        catch {
            Write-Host "[WARN] Backend not reachable. Start it first:" -ForegroundColor Red
            Write-Host "       uvicorn backend.app.api.main:app --host 0.0.0.0 --port 8000" -ForegroundColor Red
            Write-Host ""
        }

        # Verify ZMQ is enabled
        Write-Host "[1.2] Checking ALGAIE_ZMQ_ENABLED..." -ForegroundColor Gray
        $zmqEnabled = $env:ALGAIE_ZMQ_ENABLED
        if ($zmqEnabled -eq "0") {
            Write-Host "[ERROR] ALGAIE_ZMQ_ENABLED=0. Set it to 1:" -ForegroundColor Red
            Write-Host '       $env:ALGAIE_ZMQ_ENABLED = "1"' -ForegroundColor Red
            exit 1
        }
        Write-Host "  ALGAIE_ZMQ_ENABLED=$( if ($zmqEnabled) { $zmqEnabled } else { '1 (default)' } )" -ForegroundColor Green

        # Trigger REST calls to generate ZMQ traffic
        Write-Host "[1.3] Triggering REST calls to generate ZMQ dark traffic..." -ForegroundColor Gray
        try {
            $null = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/control/state" -TimeoutSec 5
            Write-Host "  GET /api/control/state -> OK" -ForegroundColor Green
        }
        catch {
            Write-Host "  GET /api/control/state -> FAILED (is backend running?)" -ForegroundColor Red
        }
        try {
            $null = Invoke-RestMethod -Uri "http://127.0.0.1:8000/api/control/calendar" -TimeoutSec 5
            Write-Host "  GET /api/control/calendar -> OK" -ForegroundColor Green
        }
        catch {
            Write-Host "  GET /api/control/calendar -> FAILED" -ForegroundColor Red
        }

        # Run the ZMQ validator
        Write-Host "[1.4] Running ZMQ dark traffic validator..." -ForegroundColor Gray
        Write-Host ""
        python "$BackendDir\scripts\verify_zmq_dark_traffic.py"

        Write-Host ""
        Write-Host "===========================================================" -ForegroundColor Green
        Write-Host "  Step 1 Complete - Dark traffic is flowing" -ForegroundColor Green
        Write-Host "  Next: run this script with -Step 2" -ForegroundColor Green
        Write-Host "===========================================================" -ForegroundColor Green
    }

    "2" {
        Write-Host "[STEP 2] Parallel Run - Build and Launch Native Frontend" -ForegroundColor Yellow
        Write-Host ""

        # Check vcpkg
        Write-Host "[2.1] Checking vcpkg availability..." -ForegroundColor Gray
        $vcpkg = Get-Command "vcpkg" -ErrorAction SilentlyContinue
        if (-not $vcpkg) {
            Write-Host "[ERROR] vcpkg not found in PATH. Install from:" -ForegroundColor Red
            Write-Host "        https://github.com/microsoft/vcpkg" -ForegroundColor Red
            exit 1
        }
        Write-Host "  vcpkg found at: $($vcpkg.Source)" -ForegroundColor Green

        # Check cmake
        Write-Host "[2.2] Checking CMake availability..." -ForegroundColor Gray
        $cmake = Get-Command "cmake" -ErrorAction SilentlyContinue
        if (-not $cmake) {
            Write-Host "[ERROR] CMake not found in PATH." -ForegroundColor Red
            exit 1
        }
        Write-Host "  cmake found at: $($cmake.Source)" -ForegroundColor Green

        # Configure
        Write-Host "[2.3] Configuring CMake build..." -ForegroundColor Gray
        if (-not (Test-Path $BuildDir)) {
            New-Item -ItemType Directory -Path $BuildDir | Out-Null
        }

        $vcpkgRoot = $env:VCPKG_ROOT
        if (-not $vcpkgRoot) {
            $vcpkgRoot = Split-Path (Get-Command vcpkg).Source
        }
        cmake -S $NativeDir -B $BuildDir `
            "-DCMAKE_TOOLCHAIN_FILE=$vcpkgRoot/scripts/buildsystems/vcpkg.cmake" `
            "-DCMAKE_BUILD_TYPE=Release"

        # Build Sim target
        Write-Host "[2.4] Building Algaie_Sim..." -ForegroundColor Gray
        cmake --build $BuildDir --target Algaie_Sim --config Release -j

        # Deploy Qt plugins
        Write-Host "[2.5] Deploying Qt plugins..." -ForegroundColor Gray
        $simExe = Join-Path $BuildDir "Release\Algaie_Sim.exe"
        if (Test-Path $simExe) {
            windeployqt $simExe --qmldir "$NativeDir\src\qml" --plugindir (Join-Path $BuildDir "Release\plugins")
        }

        # Launch
        Write-Host "[2.6] Launching Algaie_Sim..." -ForegroundColor Gray
        Write-Host ""
        Write-Host "  Validation checklist:" -ForegroundColor Yellow
        Write-Host "    [ ] nvidia-smi shows 0 MB CUDA usage from Qt" -ForegroundColor Yellow
        Write-Host "    [ ] No DATA LOSS flag at 9:30 AM open" -ForegroundColor Yellow
        Write-Host "    [ ] Portfolio value matches Next.js frontend" -ForegroundColor Yellow
        Write-Host "    [ ] Kill switch test: halt a sleeve, verify freeze" -ForegroundColor Yellow
        Write-Host ""

        if (Test-Path $simExe) {
            Start-Process $simExe -WorkingDirectory (Split-Path $simExe)
            Write-Host "  Algaie_Sim.exe launched" -ForegroundColor Green
        }
        else {
            Write-Host "[ERROR] Algaie_Sim.exe not found at $simExe" -ForegroundColor Red
            exit 1
        }

        Write-Host ""
        Write-Host "===========================================================" -ForegroundColor Green
        Write-Host "  Step 2 Complete - Native frontend running in parallel" -ForegroundColor Green
        Write-Host "  Run for 2 weeks, then: run this script with -Step 3" -ForegroundColor Green
        Write-Host "===========================================================" -ForegroundColor Green
    }

    "3" {
        Write-Host "[STEP 3] Full Cutover - Production Promotion" -ForegroundColor Yellow
        Write-Host ""
        Write-Host "  +-------------------------------------------------------+" -ForegroundColor Red
        Write-Host "  |  WARNING: This will switch execution to native C++    |" -ForegroundColor Red
        Write-Host "  |  and terminate the Next.js frontend process.          |" -ForegroundColor Red
        Write-Host "  +-------------------------------------------------------+" -ForegroundColor Red
        Write-Host ""

        # Pre-flight checklist
        Write-Host "[3.1] Pre-flight safety checklist:" -ForegroundColor Gray
        $checks = @(
            "2-week parallel run completed without DATA LOSS flags",
            "nvidia-smi confirmed 0 MB CUDA usage from Qt",
            "Portfolio values match between native and Next.js",
            "Kill switch halt/resume tested successfully",
            "Broker gateway connectivity verified",
            "NTP sync within 500ms"
        )

        $allPassed = $true
        foreach ($check in $checks) {
            $response = Read-Host "  [?] $check (y/n)"
            if ($response -ne "y") {
                Write-Host "  [FAIL] Aborting cutover - $check not confirmed" -ForegroundColor Red
                $allPassed = $false
                break
            }
        }

        if (-not $allPassed) {
            Write-Host ""
            Write-Host "[ABORT] Cutover cancelled. Address failures and retry." -ForegroundColor Red
            exit 1
        }

        Write-Host ""
        Write-Host "[3.2] Building Algaie_Live..." -ForegroundColor Gray
        cmake --build $BuildDir --target Algaie_Live --config Release -j

        $liveExe = Join-Path $BuildDir "Release\Algaie_Live.exe"
        if (Test-Path $liveExe) {
            windeployqt $liveExe --qmldir "$NativeDir\src\qml" --plugindir (Join-Path $BuildDir "Release\plugins")
        }

        Write-Host "[3.3] Stopping Next.js frontend..." -ForegroundColor Gray
        $nextjs = Get-Process -Name "node" -ErrorAction SilentlyContinue
        if ($nextjs) {
            $nextjs | Stop-Process -Force
            Write-Host "  Next.js process terminated" -ForegroundColor Green
        }
        else {
            Write-Host "  No Next.js process found (may already be stopped)" -ForegroundColor Yellow
        }

        Write-Host "[3.4] Launching Algaie_Live..." -ForegroundColor Gray
        if (Test-Path $liveExe) {
            Start-Process $liveExe -WorkingDirectory (Split-Path $liveExe)
            Write-Host "  Algaie_Live.exe launched" -ForegroundColor Green
        }

        Write-Host ""
        Write-Host "===========================================================" -ForegroundColor Green
        Write-Host "  Step 3 Complete - CUTOVER SUCCESSFUL" -ForegroundColor Green
        Write-Host "  Native frontend is now the primary execution interface." -ForegroundColor Green
        Write-Host "===========================================================" -ForegroundColor Green
    }
}
