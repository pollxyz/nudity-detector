"""
Downloads the larger NudeNet model (640m.onnx, ~104 MB) used by the detector.

The model is hosted on the NudeNet author's GitHub Releases. The browser
download URL redirects to a GitHub login page (the repo content is gated),
so we use the GitHub REST API asset endpoint with the right Accept header,
which returns the file directly.

Idempotent — skips download if the file already exists and looks valid.
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = PROJECT_ROOT / "models_640m.onnx"
EXPECTED_SIZE = 103_538_690  # bytes — from the upstream release manifest
ASSET_API_URL = "https://api.github.com/repos/notAI-tech/NudeNet/releases/assets/176832019"


def _is_valid(path: Path) -> bool:
    if not path.exists():
        return False
    actual = path.stat().st_size
    # Allow small slop for line-ending or other transfer differences
    return abs(actual - EXPECTED_SIZE) < 1024


def download() -> int:
    if _is_valid(MODEL_PATH):
        size_mb = MODEL_PATH.stat().st_size / 1024 / 1024
        print(f"[ok] {MODEL_PATH.name} already present ({size_mb:.1f} MB), skipping download.")
        return 0

    print(f"Downloading NudeNet 640m model (~104 MB)...")
    print(f"  to: {MODEL_PATH}")
    req = urllib.request.Request(
        ASSET_API_URL,
        headers={
            "User-Agent": "NudityDetection/1.0",
            "Accept": "application/octet-stream",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response, MODEL_PATH.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            written = 0
            chunk_mb = 0
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                written += len(chunk)
                chunk_mb += 1
                if total > 0:
                    pct = written * 100 / total
                    sys.stdout.write(f"\r  {written/1024/1024:6.1f} MB / {total/1024/1024:.1f} MB  ({pct:5.1f}%)")
                else:
                    sys.stdout.write(f"\r  {written/1024/1024:6.1f} MB")
                sys.stdout.flush()
        print()
    except urllib.error.URLError as exc:
        print(f"\n[error] download failed: {exc}", file=sys.stderr)
        # Don't leave a partial file behind — confuses the wrapper later
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        return 1

    if not _is_valid(MODEL_PATH):
        print(f"[error] downloaded file size mismatch", file=sys.stderr)
        return 2

    print(f"[ok] saved {MODEL_PATH.name} ({MODEL_PATH.stat().st_size/1024/1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(download())
