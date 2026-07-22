"""
REST API for the Bangladeshi Taka note detector (Phase-2, Task 2).

FastAPI app exposing:
    GET  /            -> service metadata
    GET  /health      -> liveness/readiness probe (used by Docker healthcheck)
    POST /predict     -> detect denominations in an uploaded image

The model is loaded once at startup and shared across requests. Invalid or
missing inputs return descriptive JSON errors with appropriate HTTP status
codes rather than a 500 stack trace.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app import config
from app.inference import InvalidImageError, TakaDetector

# Populated at startup; module-level so every request reuses one loaded model.
detector: TakaDetector | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the process starts."""
    global detector
    detector = TakaDetector()
    yield
    detector = None


app = FastAPI(
    title="Bangladeshi Taka Note Detection API",
    description="Detects Taka denominations in an image using a YOLO11 model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/")
def root():
    """Basic service metadata and usage hint."""
    return {
        "service": "Bangladeshi Taka Note Detection API",
        "model_loaded": detector is not None,
        "classes": list(detector.class_names.values()) if detector else [],
        "usage": "POST an image file (JPEG/PNG) to /predict",
    }


@app.get("/health")
def health():
    """Readiness probe — 200 only once the model is loaded."""
    if detector is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Detect Taka denominations in an uploaded image.

    Returns detected denomination names, confidence scores, and bounding-box
    coordinates. Rejects non-image and oversized uploads with 4xx errors.
    """
    if detector is None:  # pragma: no cover - startup guard
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # 1. Validate the declared content type.
    if file.content_type not in config.ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type '{file.content_type}'. "
                f"Send one of: {', '.join(sorted(config.ALLOWED_CONTENT_TYPES))}."
            ),
        )

    # 2. Read the payload and enforce the size limit.
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    max_bytes = int(config.MAX_UPLOAD_MB * 1024 * 1024)
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(raw) / 1e6:.1f} MB). Limit is {config.MAX_UPLOAD_MB} MB.",
        )

    # 3. Run inference; a non-decodable image is the client's fault (400).
    try:
        result = detector.predict(raw)
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return JSONResponse(
        content={
            "filename": file.filename,
            "detections": result["detections"],
            "count": result["count"],
            "image_size": result["image_size"],
        }
    )
