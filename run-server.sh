#!/usr/bin/env bash
# Launches the moderation HTTP API for PollXYZ to call.
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/activate" ]; then
    echo "Virtual environment missing. Run ./install.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
echo "Starting moderation API on http://0.0.0.0:8000"
echo "Health check:  http://127.0.0.1:8000/health"
echo "Active policy: http://127.0.0.1:8000/policy"
echo "Docs:          http://127.0.0.1:8000/docs"
echo
python -m uvicorn server:app --host 0.0.0.0 --port 8000 "$@"
