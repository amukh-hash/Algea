@echo off
:: Algae Dashboard Launcher
:: Starts Backend API, Orchestrator, Frontend, and opens the Browser.

cd /d "%~dp0.."
TITLE Algae Launcher v4.0

echo [1/3] Launching Backend API & Orchestrator...
start "Algae Backend" powershell -NoExit -ExecutionPolicy Bypass -File "scripts\start_trading.ps1"

echo [2/3] Launching Frontend (Next.js)...
cd frontend
start "Algae Frontend" npm run dev

echo [3/3] Opening Dashboard...
timeout /t 15 >nul
start http://localhost:3000/execution

echo Done. You can minify this window.
timeout /t 3 >nul
exit
