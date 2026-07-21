"""
Evaluates a trained model against the validation/test split and prints
overall + per-class metrics.

Usage:
    python3 src/evaluate.py --weights runs/detect/train/weights/best.pt --data dataset/data.yaml
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLO11 Taka-detection model.")
    parser.add_argument("--weights", required=True, help="Path to trained best.pt")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="Which split to evaluate on")
    args = parser.parse_args()

    model = YOLO(args.weights)
    metrics = model.val(data=args.data, split=args.split)

    print(f"\n=== Evaluation on '{args.split}' split ===")
    print(f"mAP50:     {metrics.box.map50:.4f}")
    print(f"mAP50-95:  {metrics.box.map:.4f}")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall:    {metrics.box.mr:.4f}")

    print("\nPer-class mAP50-95:")
    for class_id, class_name in model.names.items():
        try:
            print(f"  {class_name:20s} {metrics.box.maps[class_id]:.4f}")
        except (IndexError, KeyError):
            continue


if __name__ == "__main__":
    main()
