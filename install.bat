@echo off
REM One-time setup. Creates a local virtual environment and installs dependencies.

setlocal
cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python is not on PATH. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo Downloading detection model (~104 MB, one-time)...
python download_models.py
if errorlevel 1 (
    echo WARNING: model download failed. The app will fall back to the
    echo smaller / less accurate model bundled with the nudenet pip package.
)

echo.
echo ============================================================
echo  Setup complete.
echo  Double-click run.bat to launch the web UI.
echo  Double-click run-server.bat to launch the moderation API.
echo  Double-click run-share.bat for a public share link.
echo ============================================================
pause
