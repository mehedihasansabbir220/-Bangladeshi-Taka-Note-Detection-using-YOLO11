# 🇧🇩 Bangladeshi Taka Note Detection — REST API & Docker (Phase-2)

Deploys the YOLO11 Taka-note detector trained in Phase-1 as a containerised REST API. Upload an image, get back the detected denominations, confidence scores, and bounding boxes as JSON.

This directory is self-contained: model weights, API code, tests, and Docker config all live here.

---

## 📂 Project Structure

```
deployment/
├── app/
│   ├── __init__.py
│   ├── config.py          # Env-overridable settings (weights path, thresholds, limits)
│   ├── inference.py        # TakaDetector — loads weights once, runs single-image inference
│   └── main.py             # FastAPI app: /, /health, /predict
├── models/
│   └── best.pt             # Trained YOLO11 weights (copied from Phase-1)
├── scripts/
│   └── infer_demo.py       # Task 1: standalone inference demo + annotated output
├── tests/
│   ├── test_api.py         # Task 3: API test harness (pytest + standalone report)
│   └── sample_images/      # 5 held-out test images
├── outputs/                # Annotated demo images land here
├── docs/
│   └── SUBMISSION.md       # Where to place screenshots / links for grading
├── Dockerfile              # Task 4: container definition
├── requirements.txt
├── .dockerignore
└── README.md               # This file
```

---

## 🧩 Class Labels

The model detects 9 denominations (class id → name):

| 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| ৳1 | ৳2 | ৳5 | ৳10 | ৳20 | ৳50 | ৳100 | ৳500 | ৳1000 |

It detects the **printed denomination numerals** on each note (not the whole banknote outline), which is how the dataset was annotated. `class_name` in the API response carries the human-readable denomination.

---

## 🚀 Quick Start (local, no Docker)

From inside `deployment/`:

```bash
# 1. Install dependencies (a virtualenv is recommended)
pip install -r requirements.txt

# 2. Make sure the weights are in place
ls models/best.pt

# 3a. Task 1 — run the single-image inference demo
python3 scripts/infer_demo.py --image tests/sample_images/sample1_1000taka.jpg

# 3b. Task 2 — start the REST API
uvicorn app.main:app --host 0.0.0.0 --port 8000
# Interactive docs at http://localhost:8000/docs
```

---

## 🔌 API Reference

### `POST /predict`

| | |
|---|---|
| **Method** | `POST` |
| **Input** | multipart form field `file` — a JPEG or PNG image |
| **Success** | `200 OK` with the JSON below |

**Example request (curl):**

```bash
curl -X POST http://localhost:8000/predict \
     -F "file=@tests/sample_images/sample2_500taka.jpg"
```

**Example response:**

```json
{
  "filename": "sample2_500taka.jpg",
  "detections": [
    {
      "class_id": 7,
      "class_name": "500 taka",
      "confidence": 0.91,
      "bbox": { "x1": 342.5, "y1": 88.0, "x2": 470.1, "y2": 210.7 }
    }
  ],
  "count": 1,
  "image_size": { "width": 640, "height": 640 }
}
```

Bounding boxes are absolute pixel coordinates in the uploaded image: `(x1, y1)` top-left, `(x2, y2)` bottom-right. Detections are sorted most-confident first.

### Error responses

| Status | When |
|---|---|
| `400 Bad Request` | Empty upload, or a file that can't be decoded as an image |
| `413 Payload Too Large` | Upload exceeds `MAX_UPLOAD_MB` (default 10 MB) |
| `415 Unsupported Media Type` | Content type isn't JPEG/PNG |
| `422 Unprocessable Entity` | The `file` field is missing entirely |
| `503 Service Unavailable` | Model hasn't finished loading |

### Other endpoints

- `GET /` — service metadata and the list of detectable classes
- `GET /health` — returns `200` once the model is loaded (used by the Docker healthcheck)
- `GET /docs` — auto-generated Swagger UI

---

## 🐳 Docker

### Build

From inside `deployment/`:

```bash
docker build -t taka-detection-api .
```

### Run

```bash
docker run --rm -p 8000:8000 taka-detection-api
```

The API is now reachable from the host at `http://localhost:8000`. Confirm it's up:

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/predict \
     -F "file=@tests/sample_images/sample1_1000taka.jpg"
```

### Useful variations

```bash
# Run detached and follow logs
docker run -d --name taka -p 8000:8000 taka-detection-api
docker logs -f taka

# Override a setting at runtime (e.g. lower the confidence threshold)
docker run --rm -p 8000:8000 -e CONF_THRESHOLD=0.15 taka-detection-api

# Swap in different weights without rebuilding
docker run --rm -p 8000:8000 -v "$PWD/models:/app/models" taka-detection-api
```

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `MODEL_WEIGHTS` | `/app/models/best.pt` | Path to the weights file |
| `CONF_THRESHOLD` | `0.25` | Minimum confidence to keep a detection |
| `IOU_THRESHOLD` | `0.45` | NMS IoU threshold |
| `IMG_SIZE` | `640` | Inference image size |
| `MAX_UPLOAD_MB` | `10` | Reject uploads larger than this |

---

## ✅ Testing (Task 3)

With the API running (locally or in Docker):

```bash
# Human-readable report over all 5 sample images + error cases (good for screenshots)
python3 tests/test_api.py --url http://localhost:8000

# Or as a pytest suite
pytest tests/test_api.py
```

---

## 📝 Notes on the model

The weights in `models/best.pt` come from the Phase-1 pipeline. Training was done on a CPU-only machine, so the shipped weights are a **short training run** — enough to demonstrate the full detection → API → Docker path end to end. For production accuracy, retrain on a GPU (e.g. Google Colab) with more epochs; see the Phase-1 README. The deployment code is model-agnostic — drop a better `best.pt` into `models/` (or mount it) and it is served unchanged.
