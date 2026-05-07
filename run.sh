#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -f ".venv/bin/activate" ]; then
    echo "Virtual environment missing. Run ./install.sh first."
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate
python app.py "$@"
