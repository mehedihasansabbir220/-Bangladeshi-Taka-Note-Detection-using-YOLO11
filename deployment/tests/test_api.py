"""
API test & validation harness (Phase-2, Task 3).

Exercises the running /predict endpoint against the 5 sample images plus a
set of error cases, and prints a summary table. Doubles as:
  - a pytest suite:      pytest tests/test_api.py
  - a standalone script:  python3 tests/test_api.py --url http://localhost:8000

The standalone mode is handy for capturing the terminal output / screenshots
the assignment asks for. It assumes the API is already running.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

import requests

DEFAULT_URL = "http://localhost:8000"
SAMPLE_DIR = Path(__file__).resolve().parent / "sample_images"


def sample_images() -> list[Path]:
    return sorted(SAMPLE_DIR.glob("sample*.*"))


def post_image(base_url: str, path: Path):
    with path.open("rb") as fh:
        return requests.post(
            f"{base_url}/predict",
            files={"file": (path.name, fh, "image/jpeg")},
            timeout=60,
        )


# ---------------------------------------------------------------------------
# Standalone runner — prints a human-readable report for screenshots.
# ---------------------------------------------------------------------------
def run_report(base_url: str) -> int:
    images = sample_images()
    if not images:
        print(f"No sample images found in {SAMPLE_DIR}", file=sys.stderr)
        return 1

    print(f"Testing API at {base_url}\n" + "=" * 60)
    ok = True
    for path in images:
        try:
            resp = post_image(base_url, path)
        except requests.RequestException as exc:
            print(f"[FAIL] {path.name}: request error: {exc}")
            ok = False
            continue

        if resp.status_code != 200:
            print(f"[FAIL] {path.name}: HTTP {resp.status_code} {resp.text}")
            ok = False
            continue

        body = resp.json()
        dets = body["detections"]
        top = ", ".join(f"{d['class_name']}({d['confidence']:.2f})" for d in dets[:4]) or "none"
        print(f"[ OK ] {path.name:<28} {body['count']} detection(s): {top}")

    print("=" * 60)
    print("\nFull JSON for the first image:\n")
    print(json.dumps(post_image(base_url, images[0]).json(), indent=2))

    # Error handling checks (Task 2 requirement).
    print("\n" + "=" * 60 + "\nError-handling checks")
    # Non-image payload sent as JPEG -> 400.
    r = requests.post(f"{base_url}/predict",
                      files={"file": ("bad.jpg", io.BytesIO(b"not an image"), "image/jpeg")}, timeout=30)
    print(f"  corrupt image      -> HTTP {r.status_code} (expect 400)")
    # Wrong content type -> 415.
    r = requests.post(f"{base_url}/predict",
                      files={"file": ("a.txt", io.BytesIO(b"hello"), "text/plain")}, timeout=30)
    print(f"  wrong content-type -> HTTP {r.status_code} (expect 415)")
    # Missing file field -> 422 (FastAPI validation).
    r = requests.post(f"{base_url}/predict", timeout=30)
    print(f"  missing file       -> HTTP {r.status_code} (expect 422)")

    return 0 if ok else 1


# ---------------------------------------------------------------------------
# pytest interface — skips cleanly if the server isn't up.
# ---------------------------------------------------------------------------
def _server_up(base_url: str) -> bool:
    try:
        return requests.get(f"{base_url}/health", timeout=5).status_code == 200
    except requests.RequestException:
        return False


def test_predict_on_samples():
    import pytest
    base_url = DEFAULT_URL
    if not _server_up(base_url):
        pytest.skip(f"API not running at {base_url}")
    for path in sample_images():
        resp = post_image(base_url, path)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert "detections" in body and "count" in body
        for det in body["detections"]:
            assert {"class_id", "class_name", "confidence", "bbox"} <= det.keys()
            assert 0.0 <= det["confidence"] <= 1.0
            b = det["bbox"]
            assert b["x2"] > b["x1"] and b["y2"] > b["y1"]


def test_rejects_corrupt_image():
    import pytest
    base_url = DEFAULT_URL
    if not _server_up(base_url):
        pytest.skip(f"API not running at {base_url}")
    r = requests.post(f"{base_url}/predict",
                      files={"file": ("bad.jpg", io.BytesIO(b"nope"), "image/jpeg")}, timeout=30)
    assert r.status_code == 400


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exercise the /predict API for screenshots.")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base URL of the running API")
    args = parser.parse_args()
    raise SystemExit(run_report(args.url.rstrip("/")))
