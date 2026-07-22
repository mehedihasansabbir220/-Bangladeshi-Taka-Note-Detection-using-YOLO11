"""
Standalone single-image inference demo (Phase-2, Task 1).

Demonstrates the inference pipeline end to end: loads the trained YOLO11
weights, runs detection on one image, prints the detected denominations /
confidences / bounding boxes, and saves an annotated copy of the image.

Usage:
    python3 scripts/infer_demo.py --image tests/sample_images/sample1.jpg
    python3 scripts/infer_demo.py --image path/to/note.jpg --out outputs/
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Make `app` importable when run from the deployment/ directory.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.inference import InvalidImageError, TakaDetector  # noqa: E402


def annotate(image_path: Path, detections: list, out_dir: Path) -> Path:
    """Draw boxes + labels onto a copy of the image and save it."""
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    for det in detections:
        b = det["bbox"]
        label = f"{det['class_name']} {det['confidence']:.2f}"
        draw.rectangle([b["x1"], b["y1"], b["x2"], b["y2"]], outline="red", width=3)
        # Label background for legibility over busy banknote textures.
        tl, tt, tr, tb = draw.textbbox((b["x1"], b["y1"]), label, font=font)
        draw.rectangle([tl, tt - (tb - tt) - 4, tr + 4, tt], fill="red")
        draw.text((tl + 2, tt - (tb - tt) - 4), label, fill="white", font=font)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{image_path.stem}_annotated.jpg"
    img.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Single-image Taka detection demo.")
    parser.add_argument("--image", required=True, help="Path to an input image (JPEG/PNG)")
    parser.add_argument("--out", default="outputs", help="Directory for the annotated image")
    parser.add_argument("--weights", default=None, help="Override path to best.pt")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.is_file():
        raise SystemExit(f"Image not found: {image_path}")

    detector = TakaDetector(weights=args.weights) if args.weights else TakaDetector()

    try:
        result = detector.predict(image_path)
    except InvalidImageError as exc:
        raise SystemExit(f"Invalid image: {exc}")

    print(f"\nImage: {image_path}  ({result['image_size']['width']}x{result['image_size']['height']})")
    print(f"Detections: {result['count']}\n")
    if not result["detections"]:
        print("  (no notes detected — try lowering CONF_THRESHOLD)")
    for i, det in enumerate(result["detections"], 1):
        b = det["bbox"]
        print(f"  {i}. {det['class_name']:<10} conf={det['confidence']:.3f}  "
              f"bbox=[{b['x1']:.0f}, {b['y1']:.0f}, {b['x2']:.0f}, {b['y2']:.0f}]")

    print("\nJSON output:")
    print(json.dumps(result, indent=2))

    annotated = annotate(image_path, result["detections"], Path(args.out))
    print(f"\nAnnotated image saved to: {annotated}")


if __name__ == "__main__":
    main()
