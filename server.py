"""
HTTP API for the moderation pipeline. PollXYZ posts an image, gets back JSON.

Endpoints:
  GET  /health         -> {"status":"ok","ready":true}
  POST /check          -> moderation verdict for one image (multipart upload)
  POST /check_url      -> moderation verdict for one image fetched from a URL
  GET  /policy         -> the active policy thresholds

Run:
  Windows:  run-server.bat
  Manual:   .venv\Scripts\python.exe -m uvicorn server:app --host 0.0.0.0 --port 8000

The first request triggers model loading (~10s). Set up a process manager
(systemd, pm2, supervisor, Windows Service) to keep this running.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image

from analyzer import CombinedAnalyzer, VerdictPolicy
from detector import EXPLICIT_CLASSES, Detector
from nsfw_classifier import NSFWClassifier


# ---------------------------------------------------------------------------
# Production policy — these are the values you "locked in" via the UI.
# Tune via env vars without code changes.
# ---------------------------------------------------------------------------
def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


PRODUCTION_RESOLUTION = _env_int("MOD_INFERENCE_RESOLUTION", 1024)
PRODUCTION_DISPLAY_THRESHOLD = _env_float("MOD_DISPLAY_THRESHOLD", 0.15)
PRODUCTION_POLICY = VerdictPolicy(
    nsfw_block_threshold=_env_float("MOD_NSFW_BLOCK", 0.70),
    nsfw_review_threshold=_env_float("MOD_NSFW_REVIEW", 0.30),
    explicit_block_threshold=_env_float("MOD_EXPLICIT_BLOCK", 0.50),
    explicit_review_threshold=_env_float("MOD_EXPLICIT_REVIEW", 0.20),
    male_score_boost=_env_float("MOD_MALE_BOOST", 2.5),
)
MAX_IMAGE_BYTES = _env_int("MOD_MAX_IMAGE_BYTES", 15 * 1024 * 1024)  # 15 MB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("moderation")


# ---------------------------------------------------------------------------
# Lifespan: load models once at process startup, not per request.
# ---------------------------------------------------------------------------
analyzer: CombinedAnalyzer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global analyzer
    log.info("Loading detector (NudeNet 640m) at resolution=%d", PRODUCTION_RESOLUTION)
    detector = Detector(
        inference_resolution=PRODUCTION_RESOLUTION,
        score_threshold=0.05,
        nms_threshold=0.05,
    )
    log.info("Loading NSFW classifier (Falconsai)")
    classifier = NSFWClassifier()
    analyzer = CombinedAnalyzer(
        detector=detector, classifier=classifier, policy=PRODUCTION_POLICY,
    )
    log.info("Ready.")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="PollXYZ Moderation Service",
    version="1.0",
    lifespan=lifespan,
)

# Adjust origins to your PollXYZ domain in production for browser-based callers.
# Default is permissive because most callers are server-side.
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("MOD_ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class DetectionDTO(BaseModel):
    cls: str = Field(..., alias="class")
    category: str
    score: float
    box: list[int]

    model_config = {"populate_by_name": True}


class CheckResponse(BaseModel):
    verdict: str  # ALLOW | REVIEW | BLOCK
    reason: str
    nsfw_score: float
    safe_score: float
    explicit_detections: list[DetectionDTO]
    all_detections: list[DetectionDTO]
    conflicts: list[str]
    elapsed_ms: int
    model_versions: dict[str, str]


class HealthResponse(BaseModel):
    status: str
    ready: bool
    inference_resolution: int


class CheckUrlBody(BaseModel):
    url: str


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------
def _ensure_ready() -> CombinedAnalyzer:
    if analyzer is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")
    return analyzer


def _validate_image_bytes(data: bytes) -> Image.Image:
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="Empty image body")
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit",
        )
    try:
        img = Image.open(io.BytesIO(data))
        img.verify()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc
    # verify() exhausts the stream — re-open for actual use
    return Image.open(io.BytesIO(data)).convert("RGB")


def _analyze_pil(img: Image.Image) -> CheckResponse:
    a = _ensure_ready()
    t0 = time.time()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img.save(tmp, format="JPEG", quality=92)
        tmp_path = tmp.name
    try:
        result = a.analyze(tmp_path)
    finally:
        Path(tmp_path).unlink(missing_ok=True)
    elapsed_ms = int((time.time() - t0) * 1000)

    explicit = [
        DetectionDTO(
            **{"class": d.class_name},
            category=d.category,
            score=round(d.score, 4),
            box=list(d.box),
        )
        for d in result.kept_detections
        if d.class_name in EXPLICIT_CLASSES and d.score >= PRODUCTION_DISPLAY_THRESHOLD
    ]
    all_dets = [
        DetectionDTO(
            **{"class": d.class_name},
            category=d.category,
            score=round(d.score, 4),
            box=list(d.box),
        )
        for d in result.kept_detections
        if d.score >= PRODUCTION_DISPLAY_THRESHOLD
    ]

    return CheckResponse(
        verdict=result.verdict,
        reason=result.reason,
        nsfw_score=round(result.nsfw_score.nsfw, 4),
        safe_score=round(result.nsfw_score.normal, 4),
        explicit_detections=explicit,
        all_detections=all_dets,
        conflicts=result.conflicts,
        elapsed_ms=elapsed_ms,
        model_versions={
            "detector": "nudenet-640m",
            "classifier": "Falconsai/nsfw_image_detection",
        },
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        ready=analyzer is not None,
        inference_resolution=PRODUCTION_RESOLUTION,
    )


@app.get("/policy")
def policy() -> dict:
    return {
        "inference_resolution": PRODUCTION_RESOLUTION,
        "display_threshold": PRODUCTION_DISPLAY_THRESHOLD,
        "max_image_bytes": MAX_IMAGE_BYTES,
        "verdict_policy": {
            "nsfw_block_threshold": PRODUCTION_POLICY.nsfw_block_threshold,
            "nsfw_review_threshold": PRODUCTION_POLICY.nsfw_review_threshold,
            "explicit_block_threshold": PRODUCTION_POLICY.explicit_block_threshold,
            "explicit_review_threshold": PRODUCTION_POLICY.explicit_review_threshold,
            "male_score_boost": PRODUCTION_POLICY.male_score_boost,
        },
    }


@app.post("/check", response_model=CheckResponse)
async def check(file: UploadFile = File(...)) -> CheckResponse:
    """
    Multipart upload:  field name = `file`.
    Returns moderation verdict + scores + detections.
    """
    data = await file.read()
    img = _validate_image_bytes(data)
    return _analyze_pil(img)


@app.post("/check_url", response_model=CheckResponse)
async def check_url(body: CheckUrlBody) -> CheckResponse:
    """
    Fetch image from a URL, then check. Useful when PollXYZ stores uploads in S3
    and wants to send the URL rather than re-upload the bytes.
    """
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(body.url)
            resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=400, detail=f"Could not fetch URL: {exc}") from exc
    img = _validate_image_bytes(resp.content)
    return _analyze_pil(img)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
