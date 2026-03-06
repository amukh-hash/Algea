@echo off
title Algae Backend
set PYTHONPATH=C:\Users\crick\ResolveLabs\Algae
cd /d C:\Users\crick\ResolveLabs\Algae
python -m uvicorn backend.app.api.main:app --host 127.0.0.1 --port 8000
pause
