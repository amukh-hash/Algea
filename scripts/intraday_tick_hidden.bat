@echo off
:: Hidden wrapper for the intraday orchestrator tick.
:: Called by Windows Task Scheduler with -WindowStyle Hidden.
cd /d "C:\Users\crick\ResolveLabs\Algae"
call run_orchestrator.bat --intraday
