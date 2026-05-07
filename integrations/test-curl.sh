#!/usr/bin/env bash
# Quick smoke test of the moderation API. Use this from any shell to verify
# the service is up before integrating with your application.

set -euo pipefail

HOST="${MOD_HOST:-127.0.0.1}"
PORT="${MOD_PORT:-8000}"
BASE="http://${HOST}:${PORT}"

if [ $# -lt 1 ]; then
    echo "Usage: $0 <path-to-image>"
    exit 1
fi

IMG="$1"
if [ ! -f "$IMG" ]; then
    echo "File not found: $IMG"
    exit 1
fi

echo "=== Health check ==="
curl -sf "$BASE/health"
echo
echo

echo "=== Active policy ==="
curl -sf "$BASE/policy"
echo
echo

echo "=== Checking $IMG ==="
curl -sf -X POST "$BASE/check" -F "file=@${IMG}"
echo
