@echo off
:: boot_tws.bat — Launch TWS via IBC and verify port 7497
:: Registered as: Algae_TWS_Boot (06:25 AM M-F)
::
:: Launches TWS through IBC for automated paper trading login,
:: then polls port 7497 for up to 3 minutes to confirm the API is ready.

cd /d "%~dp0\.."
echo [%time%] Starting TWS via IBC...

:: Check if TWS/Java is already running
powershell -Command "if (Get-Process -Name 'java','javaw' -ErrorAction SilentlyContinue) { exit 0 } else { exit 1 }"
if not errorlevel 1 (
    echo [%time%] TWS appears to already be running. Verifying port...
    goto CHECK_PORT
)

:: Launch TWS via IBC (inline mode for Task Scheduler)
call "C:\IBC\StartTWS.bat" /INLINE

:CHECK_PORT
echo [%time%] Waiting for TWS API port 7497 to become ready...
set /a RETRIES=0

:WAIT_PORT
if %RETRIES% geq 36 (
    echo [%time%] ERROR: Port 7497 not ready after 3 minutes. TWS may have failed to start.
    echo [%time%] Check C:\IBC\Logs for IBC diagnostics.
    exit /b 1
)

powershell -Command "if (Test-NetConnection 127.0.0.1 -Port 7497 -InformationLevel Quiet) { exit 0 } else { exit 1 }"
if errorlevel 1 (
    echo [%time%] Port 7497 not ready. Waiting 5 seconds... ^(Attempt %RETRIES%/36^)
    timeout /t 5 /nobreak > NUL
    set /a RETRIES+=1
    goto WAIT_PORT
)

echo [%time%] TWS API is READY on port 7497. Ready for orchestrator ignition.
exit /b 0
