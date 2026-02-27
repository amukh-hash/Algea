@echo off
REM ============================================================
REM  Algea Post-Migration Cleanup Script
REM  Run from the Algea project root on your Windows machine
REM ============================================================

echo.
echo ========================================
echo   Algea Migration Cleanup
echo ========================================
echo.

REM 1. Remove broken .venv (was created under old Aishik user)
echo [1/3] Removing broken .venv ...
if exist ".venv" (
    rmdir /s /q .venv
    echo       Done.
) else (
    echo       Already removed.
)

REM 2. Remove stale test output logs
echo [2/3] Cleaning stale log files ...
del /q pack_test_out.txt 2>nul
del /q phase2_tests.txt 2>nul
del /q full_regression.txt 2>nul
del /q test_out.txt 2>nul
del /q test_out2.txt 2>nul
del /q test_output.txt 2>nul
del /q all_tests.txt 2>nul
del /q debug.txt 2>nul
del /q debug_universe.txt 2>nul
del /q scaler.joblib 2>nul
echo       Done.

REM 3. Recreate .venv with current Python
echo [3/3] Recreating .venv ...
python -m venv .venv
if %ERRORLEVEL% NEQ 0 (
    echo       ERROR: python -m venv failed. Is Python 3.11+ on your PATH?
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
pip install -r requirements.txt
pip install -r backend\requirements.txt
echo       Done.

echo.
echo ========================================
echo   Cleanup complete!
echo ========================================
echo.
echo Next steps:
echo   1. Activate with: .venv\Scripts\activate
echo   2. Run tests with: pytest
echo   3. GPU workloads now target cuda:1 (3090 Ti)
echo      Override with: set ALGAIE_CUDA_DEVICE=cuda:0
echo.
pause
