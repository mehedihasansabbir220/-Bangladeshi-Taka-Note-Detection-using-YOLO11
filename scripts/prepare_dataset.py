"""
Turns the raw Roboflow download into a trainable dataset.

The raw export has three problems that make honest evaluation impossible:

  1. No val/test split. Everything sits in `train/`, and the shipped data.yaml
     points `val` at the training images — so any metric it reports is the
     model grading its own homework.
  2. Augmented duplicates. Roboflow emitted 2-3 augmented copies of each of the
     1,166 source photos (`1000_101_jpg.rf.<hash>.jpg`). A naive random split
     scatters copies of the same photo across train and val, leaking the answer
     and inflating mAP.
  3. Messy classes. `500 taka` (id 0) and `Five Hundred taka` (id 2) are the
     same denomination, and `currency` (id 8) is a different annotation concept
     entirely -- it boxes the whole banknote, while every other class boxes the
     printed denomination numerals. It also appears only on 500-taka photos.

This script fixes all three: it merges the duplicate 500 classes, drops
`currency`, renumbers the remaining 9 classes in denomination order, and splits
by *source photo* so augmented copies always land in the same split.

The raw download is left untouched; output goes to a separate directory.

Usage:
    python3 scripts/prepare_dataset.py
    python3 scripts/prepare_dataset.py --src dataset/train --out dataset/prepared
"""

import argparse
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

# Raw Roboflow class ids -> canonical denomination. Ids not listed are dropped.
#   0 '500 taka' and 2 'Five Hundred taka' -> same denomination, merged.
#   8 'currency' -> whole-note boxes, a different annotation concept. Dropped.
REMAP = {
    4: "1 taka",
    10: "2 taka",
    3: "5 taka",
    6: "10 taka",
    7: "20 taka",
    1: "50 taka",
    9: "100 taka",
    0: "500 taka",
    2: "500 taka",
    5: "1000 taka",
}
DROPPED = {8: "currency"}

# Final class order: ascending by denomination, so ids are stable and readable.
CLASSES = ["1 taka", "2 taka", "5 taka", "10 taka", "20 taka",
           "50 taka", "100 taka", "500 taka", "1000 taka"]
CLASS_ID = {name: i for i, name in enumerate(CLASSES)}

# '1000_101_jpg.rf.328e8de2...' -> source photo '1000_101'
STEM_RE = re.compile(r"^(.*?)_jpg\.rf\.[0-9a-f]+$")


def source_photo(stem: str) -> str:
    """The original photo a (possibly augmented) file came from."""
    match = STEM_RE.match(stem)
    return match.group(1) if match else stem


def denomination(stem: str) -> str:
    """Leading token of the filename, e.g. '1000_101' -> '1000'. Used to stratify."""
    return stem.split("_")[0]


def remap_label(text: str) -> tuple[list[str], Counter, int]:
    """Rewrite one label file's rows to the new class ids."""
    rows, kept, dropped = [], Counter(), 0
    for line in text.splitlines():
        if not line.strip():
            continue
        parts = line.split()
        old_id = int(parts[0])
        if old_id in DROPPED or old_id not in REMAP:
            dropped += 1
            continue
        new_id = CLASS_ID[REMAP[old_id]]
        rows.append(" ".join([str(new_id), *parts[1:]]))
        kept[new_id] += 1
    return rows, kept, dropped


def main():
    parser = argparse.ArgumentParser(description="Split and clean the raw Taka dataset.")
    parser.add_argument("--src", default="dataset/train", help="Raw Roboflow split to read")
    parser.add_argument("--out", default="dataset/prepared", help="Where to write train/valid/test")
    parser.add_argument("--val", type=float, default=0.15, help="Validation fraction")
    parser.add_argument("--test", type=float, default=0.15, help="Test fraction")
    parser.add_argument("--seed", type=int, default=42, help="Split seed (keep fixed for reproducibility)")
    args = parser.parse_args()

    src = Path(args.src)
    src_images, src_labels = src / "images", src / "labels"
    if not src_images.is_dir():
        raise SystemExit(f"No images at {src_images}. Run scripts/download_dataset.py first.")

    # Group every file by the source photo it was augmented from.
    groups = defaultdict(list)
    for image in sorted(src_images.iterdir()):
        if image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        groups[source_photo(image.stem)].append(image)
    if not groups:
        raise SystemExit(f"No images found in {src_images}.")

    # Stratify by denomination so each split sees every class, then shuffle
    # within each denomination and slice. Splitting groups (not files) is what
    # keeps augmented copies of one photo out of two different splits.
    by_denomination = defaultdict(list)
    for name in groups:
        by_denomination[denomination(name)].append(name)

    rng = random.Random(args.seed)
    assigned = {}
    for denom, names in sorted(by_denomination.items()):
        names = sorted(names)
        rng.shuffle(names)
        n = len(names)
        n_test = max(1, round(n * args.test))
        n_val = max(1, round(n * args.val))
        for i, name in enumerate(names):
            if i < n_test:
                assigned[name] = "test"
            elif i < n_test + n_val:
                assigned[name] = "valid"
            else:
                assigned[name] = "train"

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "valid", "test"):
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)

    counts = {s: Counter() for s in ("train", "valid", "test")}
    files = Counter()
    photos = Counter()
    empty, total_dropped, missing = [], 0, 0

    for name, images in sorted(groups.items()):
        split = assigned[name]
        photos[split] += 1
        for image in images:
            label = src_labels / f"{image.stem}.txt"
            if not label.is_file():
                missing += 1
                continue
            rows, kept, dropped = remap_label(label.read_text())
            total_dropped += dropped
            if not rows:
                # Every box was dropped -- keeping this would teach the model
                # that a clearly visible note is background.
                empty.append(image.name)
                continue
            shutil.copy2(image, out / split / "images" / image.name)
            (out / split / "labels" / f"{image.stem}.txt").write_text("\n".join(rows) + "\n")
            counts[split].update(kept)
            files[split] += 1

    yaml_path = out / "data.yaml"
    names_block = "\n".join(f"  {i}: {n}" for i, n in enumerate(CLASSES))
    yaml_path.write_text(
        "# Generated by scripts/prepare_dataset.py -- do not edit by hand.\n"
        f"# Source: {src}  seed: {args.seed}  val: {args.val}  test: {args.test}\n"
        f"path: {out.resolve()}\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(CLASSES)}\n"
        f"names:\n{names_block}\n"
    )

    print(f"Source photos: {len(groups)}  ->  image files: {sum(files.values())}")
    print(f"Dropped {total_dropped} boxes from removed classes ({', '.join(DROPPED.values())})")
    if empty:
        print(f"Skipped {len(empty)} images left with no boxes after cleaning")
    if missing:
        print(f"Warning: {missing} images had no matching label file")

    print(f"\n{'split':>6} {'photos':>7} {'images':>7} {'boxes':>7}")
    for split in ("train", "valid", "test"):
        print(f"{split:>6} {photos[split]:>7} {files[split]:>7} {sum(counts[split].values()):>7}")

    print(f"\n{'class':>10} {'train':>7} {'valid':>7} {'test':>7}")
    for i, name in enumerate(CLASSES):
        print(f"{name:>10} {counts['train'][i]:>7} {counts['valid'][i]:>7} {counts['test'][i]:>7}")

    print(f"\nWrote {yaml_path}")
    print(f"Train with:  python3 src/train.py --data {yaml_path} --model yolo11s.pt --epochs 100")


if __name__ == "__main__":
    main()
