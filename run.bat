@echo off
REM Launches the local web UI. Open the printed URL in your browser.

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment missing. Run install.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python app.py %*
pause
