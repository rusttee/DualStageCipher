@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe check_offline_ready.py
) else (
  python check_offline_ready.py
)
pause
