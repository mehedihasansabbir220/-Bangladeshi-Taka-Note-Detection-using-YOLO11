"""
Pipeline sanity check: can the model memorise a tiny slice of the data?

Before spending hours on a 100-epoch run, train on ~50 images for a few dozen
epochs and evaluate on those *same* images. This is the one situation where
training on your validation set is correct -- the point is not to measure
generalisation but to prove the plumbing works.

A healthy pipeline scores mAP50 > 0.90 here. Read a failure by how far it fell:

  ~0.00-0.30  Structurally broken. Wrong label format, coordinates in pixels
              instead of normalised, mismatched class count, or images not
              pairing with labels. Fix before training anything.
  ~0.70-0.89  Under-fit, not broken -- the model is learning the right things
              but hasn't converged. Raise --epochs or --imgsz and re-run.

Verified on this dataset: 60 epochs at 640px reaches mAP50 0.99, while 30
epochs at 320px stalls at 0.81 -- same data, same code, just under-trained.
That gap is why the defaults below are what they are.

Takes roughly 10 minutes on a CPU-only machine.

Augmentation is disabled so the model sees identical images every epoch;
leaving it on makes memorisation harder and muddies the signal.

Usage:
    python3 scripts/overfit_check.py
    python3 scripts/overfit_check.py --images 50 --epochs 30 --model yolo11n.pt
"""

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from ultralytics import YOLO

# Turn every augmentation off -- we want the exact same pixels each epoch.
NO_AUGMENTATION = dict(
    mosaic=0.0, mixup=0.0, copy_paste=0.0, erasing=0.0,
    hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
    degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0,
    fliplr=0.0, flipud=0.0,
)

PASS_THRESHOLD = 0.90


def classes_in(label: Path) -> set:
    return {int(line.split()[0]) for line in label.read_text().splitlines() if line.strip()}


def main():
    parser = argparse.ArgumentParser(description="Verify the training pipeline can overfit a tiny subset.")
    parser.add_argument("--data", default="dataset/prepared/data.yaml", help="Prepared data.yaml")
    parser.add_argument("--images", type=int, default=50, help="How many images to memorise")
    parser.add_argument("--epochs", type=int, default=60,
                         help="Fewer than ~50 under-fits and produces a misleading FAIL")
    parser.add_argument("--model", default="yolo11n.pt", help="Small model is fine — this tests plumbing, not capacity")
    parser.add_argument("--imgsz", type=int, default=640,
                         help="Must match the size you will train at; 320 halves the numeral boxes and under-fits")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--workdir", default="dataset/overfit_check", help="Scratch dataset location")
    args = parser.parse_args()

    data_yaml = Path(args.data)
    if not data_yaml.is_file():
        raise SystemExit(f"{data_yaml} not found. Run scripts/prepare_dataset.py first.")
    cfg = yaml.safe_load(data_yaml.read_text())
    names = cfg["names"]
    names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)

    root = Path(cfg["path"])
    train_images = root / Path(cfg["train"])
    train_labels = train_images.parent / "labels"

    # Pick a subset that covers every class -- round-robin over classes so a
    # rare denomination like 500 taka is represented, not sampled away.
    by_class = defaultdict(list)
    for label in sorted(train_labels.glob("*.txt")):
        for class_id in classes_in(label):
            by_class[class_id].append(label)
    if not by_class:
        raise SystemExit(f"No labels found under {train_labels}")

    picked, seen = [], set()
    while len(picked) < args.images:
        added = False
        for class_id in sorted(by_class):
            queue = by_class[class_id]
            while queue:
                label = queue.pop(0)
                if label in seen:
                    continue
                seen.add(label)
                picked.append(label)
                added = True
                break
            if len(picked) >= args.images:
                break
        if not added:
            break  # exhausted the dataset

    work = Path(args.workdir)
    if work.exists():
        shutil.rmtree(work)
    (work / "images").mkdir(parents=True)
    (work / "labels").mkdir(parents=True)

    tally = Counter()
    for label in picked:
        image = next((p for p in train_images.glob(f"{label.stem}.*")), None)
        if image is None:
            continue
        shutil.copy2(image, work / "images" / image.name)
        shutil.copy2(label, work / "labels" / label.name)
        tally.update(classes_in(label))

    # train and val deliberately point at the same images.
    tiny_yaml = work / "data.yaml"
    tiny_yaml.write_text(yaml.safe_dump({
        "path": str(work.resolve()),
        "train": "images",
        "val": "images",
        "nc": len(names),
        "names": {i: n for i, n in enumerate(names)},
    }, sort_keys=False))

    n_images = len(list((work / "images").iterdir()))
    print(f"Overfit check: {n_images} images, {args.epochs} epochs, augmentation off")
    print("Class coverage: " + ", ".join(f"{names[c]}={tally[c]}" for c in sorted(tally)))
    missing = [names[i] for i in range(len(names)) if i not in tally]
    if missing:
        print(f"Not represented in subset: {', '.join(missing)}")
    print()

    train_kwargs = dict(
        data=str(tiny_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        project="runs/overfit_check", name="run", exist_ok=True,
        patience=0, val=False, plots=False, verbose=False, **NO_AUGMENTATION,
    )
    if args.device is not None:
        train_kwargs["device"] = args.device

    model = YOLO(args.model)
    model.train(**train_kwargs)

    metrics = model.val(data=str(tiny_yaml), imgsz=args.imgsz, split="val", verbose=False, plots=False)
    map50, map5095 = metrics.box.map50, metrics.box.map

    print("\n" + "=" * 58)
    print(f"mAP50 on the memorised images: {map50:.4f}   (mAP50-95: {map5095:.4f})")
    if map50 >= PASS_THRESHOLD:
        print(f"PASS — the pipeline can fit its data. A full run is worth starting.")
    else:
        print(f"FAIL — expected mAP50 >= {PASS_THRESHOLD:.2f}.")
        if map50 >= 0.60:
            print("Close, though — this reads as under-fit rather than broken.")
            print("Re-run with more budget before suspecting the data:")
            print(f"  python3 scripts/overfit_check.py --epochs {args.epochs * 2} --imgsz {max(args.imgsz, 640)}")
        else:
            print("This far down, the data plumbing is the suspect. Check:")
            print("  - label coords are normalised 0-1, not pixels")
            print("  - nc matches the highest class id actually present")
            print("  - every image has a matching .txt of the same stem")
    print("=" * 58)

    print("\nPer-class mAP50-95 (only classes present in the subset):")
    for i, name in enumerate(names):
        if i in tally:
            try:
                print(f"  {name:12s} {metrics.box.maps[i]:.4f}")
            except (IndexError, KeyError):
                continue


if __name__ == "__main__":
    main()
