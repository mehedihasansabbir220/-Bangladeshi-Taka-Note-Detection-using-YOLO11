"""
Inference pipeline for the Bangladeshi Taka note detector (Phase-2, Task 1).

Wraps the trained YOLO11 weights in a small, framework-agnostic class so the
same detection code is reused by the CLI demo, the REST API, and the tests.
The model is loaded once and reused across calls — loading weights per request
would dominate latency.
"""

from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

from app import config


class InvalidImageError(ValueError):
    """Raised when the supplied bytes are not a decodable image."""


class TakaDetector:
    """Loads YOLO11 weights once and runs single-image inference."""

    def __init__(self, weights: str | Path = config.MODEL_WEIGHTS,
                 conf: float = config.CONF_THRESHOLD,
                 iou: float = config.IOU_THRESHOLD,
                 imgsz: int = config.IMG_SIZE):
        weights = Path(weights)
        if not weights.is_file():
            raise FileNotFoundError(
                f"Model weights not found at '{weights}'. "
                "Copy your trained best.pt into deployment/models/ "
                "or set the MODEL_WEIGHTS environment variable."
            )
        self.weights = str(weights)
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        # Loading here (not per request) keeps inference fast and warms the
        # model at startup so the first request isn't penalised.
        self.model = YOLO(self.weights)
        # class-id -> denomination name, taken from the weights themselves.
        self.class_names = self.model.names

    def _load_image(self, image: str | Path | bytes | Image.Image) -> Image.Image:
        """Accept a path, raw bytes, or a PIL image and return RGB pixels."""
        try:
            if isinstance(image, (str, Path)):
                img = Image.open(image)
            elif isinstance(image, bytes):
                img = Image.open(io.BytesIO(image))
            elif isinstance(image, Image.Image):
                img = image
            else:
                raise InvalidImageError(f"Unsupported image type: {type(image)!r}")
            return img.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise InvalidImageError("Could not decode the supplied file as an image.") from exc

    def predict(self, image: str | Path | bytes | Image.Image) -> dict:
        """Run detection on one image and return a JSON-serialisable result.

        Returns a dict shaped like:
            {
              "detections": [
                {"class_id": 7, "class_name": "500 taka",
                 "confidence": 0.94,
                 "bbox": {"x1": .., "y1": .., "x2": .., "y2": ..}},
                ...
              ],
              "count": <number of detections>,
              "image_size": {"width": W, "height": H}
            }
        Bounding boxes are absolute pixel coordinates in the input image,
        with (x1, y1) the top-left and (x2, y2) the bottom-right corner.
        """
        img = self._load_image(image)
        width, height = img.size

        results = self.model.predict(
            source=img,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )

        detections = []
        # predict() returns one Results object per image; we sent exactly one.
        for box in results[0].boxes:
            class_id = int(box.cls)
            x1, y1, x2, y2 = (round(float(v), 2) for v in box.xyxy[0].tolist())
            detections.append({
                "class_id": class_id,
                "class_name": self.class_names[class_id],
                "confidence": round(float(box.conf), 4),
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
            })

        # Most-confident first — the natural reading order for a client.
        detections.sort(key=lambda d: d["confidence"], reverse=True)

        return {
            "detections": detections,
            "count": len(detections),
            "image_size": {"width": width, "height": height},
        }
