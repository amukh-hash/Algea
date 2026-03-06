@echo off
title Algae Trading System Launcher
color 0B
echo.
echo   ===============================
echo    ALGAE TRADING SYSTEM LAUNCHER
echo   ===============================
echo.

:: -- 1. Start Backend --
echo [1/3] Starting backend server...

:: Check if already running
curl -s -o nul -w "" http://127.0.0.1:8000/healthz >nul 2>nul
if %ERRORLEVEL% == 0 (
    echo   Backend already running - skipping
    goto :ib_connect
)

start "Algae Backend" /min "C:\Users\crick\ResolveLabs\Algae\start_backend.bat"
echo   Backend process launched

echo   Waiting for healthcheck...
set /a count=0
:healthloop
if %count% GEQ 20 goto :health_timeout
timeout /t 1 /nobreak >nul
set /a count+=1
curl -s -o nul -w "" http://127.0.0.1:8000/healthz >nul 2>nul
if %ERRORLEVEL% == 0 goto :health_ok
goto :healthloop

:health_ok
echo   Backend is healthy!
goto :ib_connect

:health_timeout
echo   [WARN] Backend healthcheck timed out - continuing anyway

:: -- 2. Connect to IB Gateway --
:ib_connect
echo.
echo [2/3] Connecting to IB Gateway...
curl -s -X POST http://127.0.0.1:8000/api/control/broker/connect -H "Content-Type: application/json" -d "{}" -o "%TEMP%\ibkr_resp.json" 2>nul
if not %ERRORLEVEL% == 0 (
    echo   Could not reach backend for IBKR connect
    goto :launch_frontend
)
type "%TEMP%\ibkr_resp.json" | findstr /C:"true" >nul 2>nul
if %ERRORLEVEL% == 0 (
    echo   IBKR connected successfully
) else (
    echo   IBKR not available - make sure IB Gateway/TWS is running on port 7497
    echo   The frontend will retry automatically on startup
)

:: -- 3. Launch Frontend --
:launch_frontend
echo.
echo [3/3] Launching native frontend...

tasklist /FI "IMAGENAME eq Algae_Sim.exe" 2>nul | find "Algae_Sim.exe" >nul
if %ERRORLEVEL% == 0 (
    echo   Algae_Sim already running
    goto :done
)

if exist "C:\Users\crick\ResolveLabs\Algae\native_frontend\build\Release\Algae_Sim.exe" (
    cd /d "C:\Users\crick\ResolveLabs\Algae\native_frontend\build\Release"
    start "" "Algae_Sim.exe"
    echo   Algae_Sim launched
) else (
    echo   [ERROR] Frontend exe not found - rebuild required
)

:done
echo.
echo   ===============================
echo    ALL SYSTEMS LAUNCHED
echo    Backend:  http://127.0.0.1:8000
echo    ZMQ:      tcp://127.0.0.1:5556
echo    Frontend: Algae [SIMULATION]
echo   ===============================
echo.
pause
