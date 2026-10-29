@echo off
cd /d "%~dp0"
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe simulation-main.py
) else (
  python simulation-main.py
)
pause
