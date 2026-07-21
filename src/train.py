"""
Trains a YOLO11 model on the Bangladeshi Taka detection dataset.

Usage:
    python3 src/train.py --data dataset/data.yaml --model yolo11s.pt --epochs 100

Run `python3 src/train.py --help` for all options.
"""

import argparse
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLO11 on the Bangladeshi Taka dataset.")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolo11s.pt",
                         help="Base checkpoint: yolo11n.pt / yolo11s.pt / yolo11m.pt / yolo11l.pt / yolo11x.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20, help="Early-stopping patience (epochs)")
    parser.add_argument("--device", default=None, help="'0' for first GPU, 'cpu' to force CPU, blank to auto-detect")
    parser.add_argument("--project", default="runs/detect", help="Output directory root")
    parser.add_argument("--name", default="train", help="Run name (subfolder under --project)")
    parser.add_argument("--fliplr", type=float, default=0.0,
                         help="Horizontal flip probability — kept at 0 by default since banknotes are orientation-sensitive")
    args = parser.parse_args()

    model = YOLO(args.model)

    train_kwargs = dict(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
        fliplr=args.fliplr,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    results = model.train(**train_kwargs)

    print("\nTraining complete.")
    print(f"Best weights saved to: {args.project}/{args.name}/weights/best.pt")


if __name__ == "__main__":
    main()
