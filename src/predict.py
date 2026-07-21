"""
Runs inference with a trained model on an image, folder of images, video,
or webcam feed.

Usage:
    # Single image or folder of images
    python3 src/predict.py --weights runs/detect/train/weights/best.pt --source path/to/images

    # Webcam
    python3 src/predict.py --weights runs/detect/train/weights/best.pt --source 0
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained YOLO11 Taka-detection model.")
    parser.add_argument("--weights", required=True, help="Path to trained best.pt")
    parser.add_argument("--source", required=True,
                         help="Image path, folder path, video path, or '0' for webcam")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS")
    parser.add_argument("--augment", action="store_true", help="Enable test-time augmentation")
    parser.add_argument("--save", action="store_true", default=True, help="Save annotated output")
    parser.add_argument("--show", action="store_true", help="Display live predictions (for webcam/video)")
    args = parser.parse_args()

    source = 0 if args.source == "0" else args.source

    model = YOLO(args.weights)
    results = model.predict(
        source=source,
        conf=args.conf,
        iou=args.iou,
        augment=args.augment,
        save=args.save,
        show=args.show,
    )

    for r in results:
        for box in r.boxes:
            class_name = model.names[int(box.cls)]
            confidence = float(box.conf)
            print(f"{class_name}: {confidence:.2f}")


if __name__ == "__main__":
    main()
