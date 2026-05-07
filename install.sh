#!/usr/bin/env bash
# One-time setup. Creates a local virtual environment and installs dependencies.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python3 >/dev/null; then
    echo "ERROR: python3 is not on PATH. Install Python 3.10+."
    exit 1
fi

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt

echo
echo "Downloading detection model (~104 MB, one-time)..."
python download_models.py || echo "WARNING: model download failed; app will fall back to bundled smaller model"

echo
echo "============================================================"
echo "  Setup complete."
echo "  Run ./run.sh         to launch the web UI."
echo "  Run ./run-server.sh  to launch the moderation API."
echo "  Run ./run-share.sh   for a public share link."
echo "============================================================"
