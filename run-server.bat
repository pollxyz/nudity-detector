@echo off
REM Launches the moderation HTTP API for PollXYZ to call.
REM Listens on 0.0.0.0:8000 by default. Tweak port via --port.

setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\activate.bat" (
    echo Virtual environment missing. Run install.bat first.
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
echo Starting moderation API on http://0.0.0.0:8000
echo Health check:  http://127.0.0.1:8000/health
echo Active policy: http://127.0.0.1:8000/policy
echo Docs:          http://127.0.0.1:8000/docs
echo.
python -m uvicorn server:app --host 0.0.0.0 --port 8000 %*
pause
