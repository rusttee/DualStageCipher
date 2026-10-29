@echo off
setlocal
cd /d "%~dp0"
echo =============================================================
echo Nature Cipher - FIRST RUN SETUP FOR OFFLINE PRESENTATION
echo =============================================================
if not exist .venv\Scripts\python.exe (
  py -m venv .venv
  if errorlevel 1 python -m venv .venv
)
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python prepare_offline.py --cache-wheels
python check_offline_ready.py
echo.
echo Setup finished. Keep this whole folder for the presentation.
pause
