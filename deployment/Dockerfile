# syntax=docker/dockerfile:1
#
# Container image for the Bangladeshi Taka Note Detection API (Phase-2, Task 4).
#
# Uses a slim Python base, installs the CPU-only inference stack, copies the
# model weights and source, exposes the API port, and serves with uvicorn.

FROM python:3.12-slim

# - PYTHONUNBUFFERED: logs stream out immediately (visible in `docker logs`)
# - PYTHONDONTWRITEBYTECODE: no .pyc clutter in the image
# - PIP_NO_CACHE_DIR: smaller image
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_WEIGHTS=/app/models/best.pt

# OpenCV (pulled in by ultralytics) needs libGL and libglib at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU-only torch wheels explicitly, then the rest of the deps.
# Doing torch first (from the CPU index) avoids pulling multi-GB CUDA wheels.
COPY requirements.txt .
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Copy application code and model weights.
COPY app/ ./app/
COPY models/ ./models/

# FastAPI/uvicorn listen here by default. PORT can be overridden at runtime
# (Google Cloud Run, for example, injects its own $PORT — usually 8080).
ENV PORT=8000
EXPOSE 8000

# Fail the container's healthcheck until the model has loaded.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python3 -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen(f\"http://localhost:{os.getenv('PORT','8000')}/health\").status==200 else 1)" || exit 1

# Shell form so ${PORT} is expanded at runtime. 0.0.0.0 makes the port
# reachable from outside the container, not just inside it.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}
