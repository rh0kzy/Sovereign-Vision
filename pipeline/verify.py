"""
sovereign-vision / pipeline / verify.py
========================================
Dataset verification, statistics, and visual sanity checks.

Checks the final dataset/ folder and reports:
- Image/label counts per split
- Class distribution per split
- Label format validation (no malformed lines, out-of-range values)
- Annotation density stats (min/max/avg boxes per image)
- Saves a visual sample grid showing random images with bounding boxes

Usage
-----
# Full verification with sample grid
python pipeline/verify.py --dataset dataset --samples 16 --output verify_output/

# Stats only (no images saved)
python pipeline/verify.py --dataset dataset --no-visuals

# Verify a specific split only
python pipeline/verify.py --dataset dataset --split train

Requirements
------------
pip install opencv-python numpy tqdm
"""

import argparse
import random
import cv2
import numpy as np
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm


# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────

SV_CLASSES = {
    0: "vehicle",
    1: "truck",
    2: "personnel",
    3: "equipment",
}

# Color per class (BGR)
CLASS_COLORS = {
    0: (0, 200, 255),    # vehicle   — yellow
    1: (0, 100, 255),    # truck     — orange
    2: (0, 255, 100),    # personnel — green
    3: (255, 80, 80),    # equipment — blue
}

SPLITS = ["train", "val", "test"]


# ─────────────────────────────────────────────
#  LABEL VALIDATION
# ─────────────────────────────────────────────

def validate_label_file(label_path: Path) -> tuple[list[dict], list[str]]:
    """
    Parse and validate a YOLO label file.

    Returns:
        (annotations, errors) where annotations is list of valid dicts
        and errors is list of error strings
    """
    annotations = []
    errors = []

    if not label_path.exists():
        errors.append(f"Missing label file: {label_path.name}")
        return annotations, errors

    lines = label_path.read_text().strip().splitlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()

        if not parts:
            continue

        if len(parts) != 5:
            errors.append(f"{label_path.name}:{i+1} — expected 5 values, got {len(parts)}")
            continue

        try:
            cls_id = int(parts[0])
            cx, cy, w, h = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"{label_path.name}:{i+1} — non-numeric values")
            continue

        # Validate class id
        if cls_id not in SV_CLASSES:
            errors.append(f"{label_path.name}:{i+1} — unknown class id {cls_id}")
            continue

        # Validate normalization
        for name, val in [("cx", cx), ("cy", cy), ("w", w), ("h", h)]:
            if val < 0 or val > 1:
                errors.append(f"{label_path.name}:{i+1} — {name}={val:.4f} out of [0,1]")

        # Validate bbox size
        if w <= 0 or h <= 0:
            errors.append(f"{label_path.name}:{i+1} — zero or negative bbox size")
            continue

        annotations.append({
            "class_id": cls_id,
            "cx": cx, "cy": cy,
            "w": w, "h": h,
        })

    return annotations, errors


# ─────────────────────────────────────────────
#  SPLIT STATISTICS
# ─────────────────────────────────────────────

def compute_split_stats(images_dir: Path, labels_dir: Path, split: str) -> dict:
    """
    Compute full statistics for one dataset split.

    Returns dict with counts, class distribution, errors, density stats.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    stats = {
        "split":          split,
        "images":         len(image_paths),
        "labels":         0,
        "empty_labels":   0,
        "missing_labels": 0,
        "errors":         [],
        "class_counts":   defaultdict(int),
        "boxes_per_image": [],
        "image_sizes":    [],
    }

    for img_path in tqdm(image_paths, desc=f"Verifying {split}", leave=False):
        lbl_path = labels_dir / (img_path.stem + ".txt")

        # Check image readability
        img = cv2.imread(str(img_path))
        if img is not None:
            h, w = img.shape[:2]
            stats["image_sizes"].append((w, h))

        # Validate label
        annotations, errors = validate_label_file(lbl_path)
        stats["errors"].extend(errors[:3])  # cap errors per file

        if not lbl_path.exists():
            stats["missing_labels"] += 1
            continue

        stats["labels"] += 1

        if not annotations:
            stats["empty_labels"] += 1
            continue

        stats["boxes_per_image"].append(len(annotations))
        for ann in annotations:
            stats["class_counts"][ann["class_id"]] += 1

    return stats


# ─────────────────────────────────────────────
#  VISUALIZATION
# ─────────────────────────────────────────────

def draw_boxes(image: np.ndarray, annotations: list[dict], thickness: int = 2) -> np.ndarray:
    """Draw YOLO bounding boxes on image."""
    result = image.copy()
    h, w = image.shape[:2]

    for ann in annotations:
        cls_id = ann["class_id"]
        cx = int(ann["cx"] * w)
        cy = int(ann["cy"] * h)
        bw = int(ann["w"] * w)
        bh = int(ann["h"] * h)

        x1 = max(0, cx - bw // 2)
        y1 = max(0, cy - bh // 2)
        x2 = min(w, cx + bw // 2)
        y2 = min(h, cy + bh // 2)

        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        cv2.rectangle(result, (x1, y1), (x2, y2), color, thickness)

        label = SV_CLASSES.get(cls_id, str(cls_id))
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(result, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(result, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    return result


def make_sample_grid(
    images_dir: Path,
    labels_dir: Path,
    n_samples: int = 16,
    cell_size: int = 300,
    seed: int = 42,
) -> np.ndarray:
    """
    Generate a grid of sample images with bounding boxes overlaid.

    Args:
        images_dir: directory containing images
        labels_dir: directory containing labels
        n_samples:  number of images to show
        cell_size:  size of each cell in the grid (pixels)
        seed:       random seed for sample selection
    Returns:
        BGR grid image
    """
    random.seed(seed)
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    if not image_paths:
        return np.zeros((cell_size, cell_size, 3), dtype=np.uint8)

    samples = random.sample(image_paths, min(n_samples, len(image_paths)))

    cols = 4
    rows = (len(samples) + cols - 1) // cols
    grid_h = rows * (cell_size + 24)
    grid_w = cols * cell_size
    grid = np.zeros((grid_h, grid_w, 3), dtype=np.uint8)

    for idx, img_path in enumerate(samples):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        lbl_path = labels_dir / (img_path.stem + ".txt")
        annotations, _ = validate_label_file(lbl_path)

        if annotations:
            img = draw_boxes(img, annotations)

        # Resize to cell
        img_resized = cv2.resize(img, (cell_size, cell_size))

        # Add filename label
        cell = np.zeros((cell_size + 24, cell_size, 3), dtype=np.uint8)
        cell[24:] = img_resized
        cv2.putText(cell, img_path.stem[:20], (4, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        row = idx // cols
        col = idx % cols
        y_start = row * (cell_size + 24)
        x_start = col * cell_size
        grid[y_start:y_start + cell_size + 24, x_start:x_start + cell_size] = cell

    return grid


# ─────────────────────────────────────────────
#  REPORT PRINTER
# ─────────────────────────────────────────────

def print_report(all_stats: list[dict]) -> None:
    """Print full verification report to console."""

    print("\n" + "═" * 56)
    print("  SOVEREIGN-VISION — DATASET VERIFICATION REPORT")
    print("═" * 56)

    total_images = 0
    total_errors = 0

    for stats in all_stats:
        split = stats["split"].upper()
        images = stats["images"]
        total_images += images
        errors = stats["errors"]
        total_errors += len(errors)
        boxes = stats["boxes_per_image"]

        print(f"\n── {split} split {'─' * (48 - len(split))}")
        print(f"  Images        : {images}")
        print(f"  Labels        : {stats['labels']}")
        print(f"  Missing labels: {stats['missing_labels']}")
        print(f"  Empty labels  : {stats['empty_labels']}")

        print(f"\n  Class distribution:")
        for cls_id, cls_name in SV_CLASSES.items():
            count = stats["class_counts"].get(cls_id, 0)
            bar_len = int(count / max(sum(stats["class_counts"].values()), 1) * 30)
            bar = "█" * bar_len
            print(f"    {cls_name:<15} {count:>6}  {bar}")

        if boxes:
            print(f"\n  Boxes per image:")
            print(f"    min={min(boxes)}  max={max(boxes)}  "
                  f"avg={sum(boxes)/len(boxes):.1f}  "
                  f"total={sum(boxes)}")

        if stats["image_sizes"]:
            widths  = [s[0] for s in stats["image_sizes"]]
            heights = [s[1] for s in stats["image_sizes"]]
            print(f"\n  Image sizes:")
            print(f"    width  min={min(widths)} max={max(widths)}")
            print(f"    height min={min(heights)} max={max(heights)}")

        if errors:
            print(f"\n  ⚠ Errors ({len(errors)} shown):")
            for e in errors[:5]:
                print(f"    {e}")
        else:
            print(f"\n  ✓ No errors found")

    print(f"\n{'═' * 56}")
    print(f"  Total images  : {total_images}")
    status = "✓ PASSED" if total_errors == 0 else f"⚠ {total_errors} ERRORS"
    print(f"  Validation    : {status}")
    print("═" * 56 + "\n")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Verify Sovereign-Vision dataset integrity and generate sample grids",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--dataset",    required=True,
                        help="Dataset root directory (contains images/ and labels/)")
    parser.add_argument("--split",      default=None, choices=SPLITS,
                        help="Verify a specific split only (default: all)")
    parser.add_argument("--samples",    type=int, default=16,
                        help="Number of sample images in visual grid (default: 16)")
    parser.add_argument("--output",     default="verify_output",
                        help="Output directory for visual grids (default: verify_output/)")
    parser.add_argument("--no-visuals", action="store_true",
                        help="Skip visual grid generation")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed for sample selection (default: 42)")
    return parser.parse_args()


def main():
    args = parse_args()

    dataset_dir = Path(args.dataset)
    output_dir  = Path(args.output)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    splits_to_check = [args.split] if args.split else SPLITS
    all_stats = []

    for split in splits_to_check:
        images_dir = dataset_dir / "images" / split
        labels_dir = dataset_dir / "labels" / split

        if not images_dir.exists():
            print(f"⚠ Skipping {split} — {images_dir} not found")
            continue

        stats = compute_split_stats(images_dir, labels_dir, split)
        all_stats.append(stats)

        # Save visual grid
        if not args.no_visuals:
            output_dir.mkdir(parents=True, exist_ok=True)
            grid = make_sample_grid(
                images_dir, labels_dir,
                n_samples=args.samples,
                seed=args.seed,
            )
            grid_path = output_dir / f"sample_grid_{split}.jpg"
            cv2.imwrite(str(grid_path), grid)
            print(f"✓ Sample grid saved → {grid_path}")

    print_report(all_stats)


if __name__ == "__main__":
    main()