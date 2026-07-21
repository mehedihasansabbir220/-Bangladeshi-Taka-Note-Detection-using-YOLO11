"""
Exports a trained best.pt to a deployment-friendly format.

Usage:
    python3 src/export.py --weights runs/detect/train/weights/best.pt --format onnx
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Export a trained YOLO11 model for deployment.")
    parser.add_argument("--weights", required=True, help="Path to trained best.pt")
    parser.add_argument("--format", default="onnx",
                         choices=["onnx", "tflite", "ncnn", "torchscript", "coreml"],
                         help="Target export format")
    args = parser.parse_args()

    model = YOLO(args.weights)
    exported_path = model.export(format=args.format)

    print(f"\nExported model saved to: {exported_path}")


if __name__ == "__main__":
    main()
