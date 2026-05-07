@echo off
REM Launches the web UI AND a public gradio.live URL valid for ~72 hours.
REM Anyone with the link can use the app — share carefully.

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment missing. Run install.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
python app.py --share
pause
