@echo off
cd /d "%~dp0"
echo [%time%] Waking up the Algae Leviathan...
echo [%time%] Working directory: %CD%

:: --- Determine port check timeout based on session type ---
:: --premarket/--eod: wait up to 5 minutes (TWS may still be booting)
:: --intraday: wait only 10 seconds (TWS should already be running)
set MAX_RETRIES=60
if /i "%~1"=="--intraday" set MAX_RETRIES=2

:: --- DISASTER RECOVERY: Wait for TWS (with graceful degradation) ---
echo [DR] Checking TWS connection on port 7497...
set /a RETRIES=0
set BROKER_AVAILABLE=0

:WAIT_PORT
if %RETRIES% geq %MAX_RETRIES% (
    echo [DR] Port 7497 not opened after %MAX_RETRIES% attempts.
    echo [DR] GRACEFUL DEGRADATION: Running orchestrator in shadow-only mode.
    echo [DR] Signals will be generated but no trades will be routed to IBKR.
    set BROKER_AVAILABLE=0
    goto RUN_ORCH
)

powershell -Command "if (Test-NetConnection 127.0.0.1 -Port 7497 -InformationLevel Quiet) { exit 0 } else { exit 1 }"
if errorlevel 1 (
    echo [DR] Port 7497 not ready. Waiting 5 seconds... ^(Attempt %RETRIES%/%MAX_RETRIES%^)
    timeout /t 5 /nobreak > NUL
    set /a RETRIES+=1
    goto WAIT_PORT
)
echo [DR] TWS is LISTENING. Full execution mode enabled.
set BROKER_AVAILABLE=1

:RUN_ORCH

:: --- EOD TIME GUARD: Prevent catch-up from executing a flatten outside 3 PM hour ---
if /i not "%~1"=="--eod" goto SKIP_EOD_GUARD
powershell -Command "if ((Get-Date).Hour -ne 15) { exit 1 } else { exit 0 }"
if errorlevel 1 (
    echo [DR] EOD Catch-Up triggered outside the 15:00 hour. Aborting to prevent after-hours execution.
    exit /b 0
)
:SKIP_EOD_GUARD

set PYTHONPATH=%~dp0
set DOTENV_PATH=%~dp0.env

call "%~dp0venv_gpu\Scripts\activate.bat"

python scripts/run_orchestrator_cycle.py %*

echo [%time%] Orchestrator Cycle Complete. BROKER_AVAILABLE=%BROKER_AVAILABLE%
