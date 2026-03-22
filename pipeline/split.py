"""
sovereign-vision / pipeline / split.py
=======================================
Stratified train/val/test split for the processed dataset.

Splits processed images+labels into the final dataset/ folder
ready for YOLOv8 training, ensuring class balance across splits.

Usage
-----
# Default 80/10/10 split
python pipeline/split.py \
  --images data/processed/images \
  --labels data/processed/labels \
  --output dataset \
  --train 0.8 --val 0.1 --test 0.1

# Custom split with fixed seed
python pipeline/split.py \
  --images data/processed/images \
  --labels data/processed/labels \
  --output dataset \
  --train 0.7 --val 0.15 --test 0.15 \
  --seed 42

Requirements
------------
pip install numpy tqdm
"""

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ─────────────────────────────────────────────
#  CLASS INFO
# ─────────────────────────────────────────────

SV_CLASSES = {
    0: "vehicle",
    1: "truck",
    2: "personnel",
    3: "equipment",
}


# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────

def read_label_classes(label_path: Path) -> list[int]:
    """Return list of class IDs present in a YOLO label file."""
    classes = []
    if not label_path.exists():
        return classes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                try:
                    classes.append(int(parts[0]))
                except ValueError:
                    continue
    return classes


def get_dominant_class(classes: list[int]) -> int:
    """Return the most frequent class in a label file (used for stratification)."""
    if not classes:
        return -1
    counts = defaultdict(int)
    for c in classes:
        counts[c] += 1
    return max(counts, key=counts.get)


def copy_files(pairs: list[tuple], images_dir: Path, labels_dir: Path,
               out_images: Path, out_labels: Path) -> int:
    """
    Copy image+label pairs to destination directories.

    Args:
        pairs:       list of (image_path, label_path) tuples
        images_dir:  source images directory
        labels_dir:  source labels directory
        out_images:  destination images directory
        out_labels:  destination labels directory
    Returns:
        number of files copied
    """
    out_images.mkdir(parents=True, exist_ok=True)
    out_labels.mkdir(parents=True, exist_ok=True)

    copied = 0
    for img_path, lbl_path in tqdm(pairs, desc=f"→ {out_images.parent.name}/{out_images.name}"):
        shutil.copy2(img_path, out_images / img_path.name)
        if lbl_path.exists():
            shutil.copy2(lbl_path, out_labels / lbl_path.name)
        else:
            # Write empty label file if missing
            (out_labels / (img_path.stem + ".txt")).touch()
        copied += 1

    return copied


# ─────────────────────────────────────────────
#  STRATIFIED SPLIT
# ─────────────────────────────────────────────

def stratified_split(
    image_label_pairs: list[tuple],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> tuple[list, list, list]:
    """
    Perform stratified split based on dominant class per image.
    Ensures each class is proportionally represented in all splits.

    Args:
        image_label_pairs: list of (image_path, label_path) tuples
        train_ratio:       fraction for training set
        val_ratio:         fraction for validation set
        test_ratio:        fraction for test set
        seed:              random seed for reproducibility
    Returns:
        (train_pairs, val_pairs, test_pairs)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    random.seed(seed)
    np.random.seed(seed)

    # Group pairs by dominant class
    class_buckets = defaultdict(list)
    for img_path, lbl_path in image_label_pairs:
        classes = read_label_classes(lbl_path)
        dominant = get_dominant_class(classes)
        class_buckets[dominant].append((img_path, lbl_path))

    train_pairs, val_pairs, test_pairs = [], [], []

    for cls_id, pairs in class_buckets.items():
        random.shuffle(pairs)
        n = len(pairs)

        n_train = int(n * train_ratio)
        n_val   = int(n * val_ratio)
        # test gets the remainder to avoid rounding loss
        n_test  = n - n_train - n_val

        train_pairs.extend(pairs[:n_train])
        val_pairs.extend(pairs[n_train:n_train + n_val])
        test_pairs.extend(pairs[n_train + n_val:])

        cls_name = SV_CLASSES.get(cls_id, f"class_{cls_id}")
        print(f"  {cls_name:<15} total={n:>4}  "
              f"train={n_train:>4}  val={n_val:>3}  test={n_test:>3}")

    # Shuffle each split so classes aren't grouped
    random.shuffle(train_pairs)
    random.shuffle(val_pairs)
    random.shuffle(test_pairs)

    return train_pairs, val_pairs, test_pairs


# ─────────────────────────────────────────────
#  DATASET YAML WRITER
# ─────────────────────────────────────────────

def write_data_yaml(output_dir: Path, class_names: dict) -> Path:
    """
    Write YOLOv8 data.yaml config file.

    Args:
        output_dir:   dataset root directory (contains images/ and labels/)
        class_names:  dict of {id: name}
    Returns:
        path to written yaml file
    """
    yaml_path = output_dir / "data.yaml"

    # Use forward slashes for cross-platform compatibility
    train_path = (output_dir / "images" / "train").as_posix()
    val_path   = (output_dir / "images" / "val").as_posix()
    test_path  = (output_dir / "images" / "test").as_posix()

    names_str = "\n".join(
        f"  {i}: {name}" for i, name in sorted(class_names.items())
    )

    yaml_content = f"""# Sovereign-Vision — YOLOv8 Dataset Config
# Generated by pipeline/split.py
# Classes: {len(class_names)}

path: {output_dir.resolve().as_posix()}

train: {train_path}
val:   {val_path}
test:  {test_path}

nc: {len(class_names)}

names:
{names_str}
"""

    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    return yaml_path


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Stratified train/val/test split for Sovereign-Vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--images",  required=True, help="Processed images directory")
    parser.add_argument("--labels",  required=True, help="Processed labels directory")
    parser.add_argument("--output",  required=True, help="Output dataset directory")
    parser.add_argument("--train",   type=float, default=0.8, help="Train ratio (default: 0.8)")
    parser.add_argument("--val",     type=float, default=0.1, help="Val ratio (default: 0.1)")
    parser.add_argument("--test",    type=float, default=0.1, help="Test ratio (default: 0.1)")
    parser.add_argument("--seed",    type=int,   default=42,  help="Random seed (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()

    images_dir = Path(args.images)
    labels_dir = Path(args.labels)
    output_dir = Path(args.output)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Validate ratios
    total = args.train + args.val + args.test
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {total:.3f}")

    # Collect all image+label pairs
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    pairs = []
    for img_path in image_paths:
        lbl_path = labels_dir / (img_path.stem + ".txt")
        pairs.append((img_path, lbl_path))

    print(f"\n── Sovereign-Vision Dataset Split ────────────────")
    print(f"  Total images : {len(pairs)}")
    print(f"  Split ratio  : train={args.train:.0%}  "
          f"val={args.val:.0%}  test={args.test:.0%}")
    print(f"  Seed         : {args.seed}")
    print(f"\n  Stratification by dominant class:")

    # Perform stratified split
    train_pairs, val_pairs, test_pairs = stratified_split(
        pairs,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
    )

    print(f"\n  Split sizes:")
    print(f"    train : {len(train_pairs)}")
    print(f"    val   : {len(val_pairs)}")
    print(f"    test  : {len(test_pairs)}")
    print()

    # Copy files to dataset/ structure
    copy_files(train_pairs,
               images_dir, labels_dir,
               output_dir / "images" / "train",
               output_dir / "labels" / "train")

    copy_files(val_pairs,
               images_dir, labels_dir,
               output_dir / "images" / "val",
               output_dir / "labels" / "val")

    copy_files(test_pairs,
               images_dir, labels_dir,
               output_dir / "images" / "test",
               output_dir / "labels" / "test")

    # Write data.yaml
    yaml_path = write_data_yaml(output_dir, SV_CLASSES)

    print(f"\n── Complete ───────────────────────────────────────")
    print(f"  dataset/images/train : {len(train_pairs)} images")
    print(f"  dataset/images/val   : {len(val_pairs)} images")
    print(f"  dataset/images/test  : {len(test_pairs)} images")
    print(f"  data.yaml written    : {yaml_path}")
    print(f"──────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()