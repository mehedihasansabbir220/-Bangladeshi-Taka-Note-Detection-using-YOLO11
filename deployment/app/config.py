"""Runtime configuration, overridable via environment variables.

Every setting has a sensible default so the app runs with zero configuration,
but each can be tuned in Docker / deployment without touching code.
"""

import os
from pathlib import Path

# Repo-relative default: deployment/models/best.pt
_DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "models" / "best.pt"

# Path to the trained YOLO11 weights. Override with MODEL_WEIGHTS=/abs/path.
MODEL_WEIGHTS = os.getenv("MODEL_WEIGHTS", str(_DEFAULT_WEIGHTS))

# Detections below this confidence are discarded before they reach the client.
CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", "0.25"))

# IoU threshold for non-maximum suppression.
IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", "0.45"))

# Inference image size — must match (or be close to) the training imgsz.
IMG_SIZE = int(os.getenv("IMG_SIZE", "640"))

# Largest upload we accept, in megabytes. Guards against oversized payloads.
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "10"))

# Image content types the /predict endpoint will accept.
ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
