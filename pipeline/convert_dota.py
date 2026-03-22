"""
sovereign-vision / pipeline / convert_dota.py
==============================================
Convert DOTA v2 dataset annotations to YOLOv8 format (normalized txt labels).

DOTA v2 annotation format (per image, one .txt file):
    x1 y1 x2 y2 x3 y3 x4 y4 category difficulty
    (4 corner points of an oriented bounding box, absolute pixel coords)

YOLOv8 output format (one .txt file per image):
    class_id cx cy w h
    (normalized 0-1, axis-aligned bounding box from OBB hull)

Class mapping (DOTA → Sovereign-Vision):
    vehicle       ← car, large-vehicle, small-vehicle, van, bus
    truck         ← truck
    personnel     ← (no direct DOTA class — skipped, added via Roboflow)
    equipment     ← storage-tank, ground-track-field, helicopter

Usage
-----
# Convert entire DOTA dataset split
python convert_dota.py --images data/raw/dota/images/train \
                       --labels data/raw/dota/labelTxt/train \
                       --output-images data/processed/images \
                       --output-labels data/processed/labels \
                       --split train

# Dry run (no files written, just prints stats)
python convert_dota.py --images data/raw/dota/images/train \
                       --labels data/raw/dota/labelTxt/train \
                       --output-images data/processed/images \
                       --output-labels data/processed/labels \
                       --dry-run

DOTA v2 Download
----------------
Register at: https://captain-whu.github.io/DOTA/dataset.html
Download: DOTA-v2.0 (images + labelTxt folders for train/val splits)

Expected DOTA folder structure:
    data/raw/dota/
    ├── images/
    │   ├── train/   ← .png images
    │   └── val/
    └── labelTxt/
        ├── train/   ← .txt annotation files
        └── val/

Requirements
------------
pip install numpy tqdm opencv-python
"""

import os
import re
import shutil
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


# ─────────────────────────────────────────────
#  CLASS MAPPING
# ─────────────────────────────────────────────

# Sovereign-Vision target classes (id → name)
SV_CLASSES = {
    0: "vehicle",
    1: "truck",
    2: "personnel",
    3: "equipment",
}

# DOTA class name → Sovereign-Vision class id
# Only listed classes are kept; all others are discarded
DOTA_TO_SV = {
    # vehicle
    "car":                0,
    "small-vehicle":      0,
    "large-vehicle":      1,
    "van":                0,
    "bus":                1,
    # truck
    "truck":              1,
    # equipment
    "storage-tank":       3,
    "helicopter":         3,
    "ground-track-field": 3,
    "harbor":             3,
    "ship":               3,
    "plane":              3,
}

# DOTA classes we intentionally ignore (speeds up processing)
DOTA_IGNORED = {
    "tennis-court", "basketball-court", "soccer-ball-field",
    "roundabout", "swimming-pool", "baseball-diamond",
    "bridge", "container-crane",
}


# ─────────────────────────────────────────────
#  CORE CONVERSION FUNCTIONS
# ─────────────────────────────────────────────

def obb_to_aabb(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    """
    Convert oriented bounding box (4 corners) to axis-aligned bounding box.

    Args:
        points: list of 4 (x, y) absolute pixel coordinate tuples
    Returns:
        (x_min, y_min, x_max, y_max) in absolute pixel coords
    """
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def aabb_to_yolo(x_min: float, y_min: float, x_max: float, y_max: float,
                 img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """
    Convert axis-aligned bounding box to normalized YOLOv8 format.

    Args:
        x_min, y_min, x_max, y_max: absolute pixel coordinates
        img_w, img_h: image dimensions
    Returns:
        (cx, cy, w, h) normalized to [0, 1]
    """
    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    w  = (x_max - x_min) / img_w
    h  = (y_max - y_min) / img_h

    # Clamp to valid range
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    w  = max(0.0, min(1.0, w))
    h  = max(0.0, min(1.0, h))

    return cx, cy, w, h


def parse_dota_label_file(label_path: Path) -> list[dict]:
    """
    Parse a DOTA v2 annotation .txt file.

    DOTA format per line:
        x1 y1 x2 y2 x3 y3 x4 y4 category difficulty

    Args:
        label_path: path to DOTA .txt annotation file
    Returns:
        list of dicts with keys: points, category, difficulty
    """
    annotations = []

    with open(label_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            # Skip header lines (DOTA files sometimes start with "imagesource:" etc.)
            if not line or line.startswith("imagesource") or line.startswith("gsd"):
                continue

            parts = line.split()
            if len(parts) < 9:
                # Malformed line — skip silently
                continue

            try:
                coords = [float(x) for x in parts[:8]]
                category = parts[8].lower().strip()
                difficulty = int(parts[9]) if len(parts) > 9 else 0

                points = [
                    (coords[0], coords[1]),
                    (coords[2], coords[3]),
                    (coords[4], coords[5]),
                    (coords[6], coords[7]),
                ]

                annotations.append({
                    "points":     points,
                    "category":   category,
                    "difficulty": difficulty,
                })

            except (ValueError, IndexError):
                # Skip malformed lines
                continue

    return annotations


def convert_dota_annotation(
    label_path: Path,
    img_w: int,
    img_h: int,
    skip_difficult: bool = False,
    min_bbox_size: int = 4,
) -> tuple[list[int], list[list[float]], dict]:
    """
    Convert a single DOTA annotation file to YOLOv8 format.

    Args:
        label_path:     path to DOTA .txt annotation file
        img_w, img_h:   image dimensions (needed for normalization)
        skip_difficult: if True, skip annotations marked as difficult (difficulty=2)
        min_bbox_size:  minimum bbox size in pixels (filter tiny boxes)
    Returns:
        (class_ids, bboxes, stats) where:
            class_ids: list of int class IDs
            bboxes:    list of [cx, cy, w, h] normalized bboxes
            stats:     dict with per-class counts and skip counts
    """
    annotations = parse_dota_label_file(label_path)

    class_ids = []
    bboxes = []
    stats = defaultdict(int)

    for ann in annotations:
        category = ann["category"]
        difficulty = ann["difficulty"]

        # Filter: skip difficult if requested
        if skip_difficult and difficulty == 2:
            stats["skipped_difficult"] += 1
            continue

        # Filter: ignore irrelevant classes
        if category in DOTA_IGNORED:
            stats["skipped_ignored"] += 1
            continue

        # Map to Sovereign-Vision class
        if category not in DOTA_TO_SV:
            stats["skipped_unknown"] += 1
            continue

        sv_class_id = DOTA_TO_SV[category]

        # Convert OBB → AABB → YOLO
        x_min, y_min, x_max, y_max = obb_to_aabb(ann["points"])

        # Filter: skip boxes that are too small (likely annotation noise)
        bbox_w_px = x_max - x_min
        bbox_h_px = y_max - y_min
        if bbox_w_px < min_bbox_size or bbox_h_px < min_bbox_size:
            stats["skipped_tiny"] += 1
            continue

        # Filter: skip boxes entirely outside image bounds
        if x_max <= 0 or y_max <= 0 or x_min >= img_w or y_min >= img_h:
            stats["skipped_oob"] += 1
            continue

        cx, cy, w, h = aabb_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h)

        class_ids.append(sv_class_id)
        bboxes.append([cx, cy, w, h])
        stats[f"kept_{SV_CLASSES[sv_class_id]}"] += 1

    return class_ids, bboxes, dict(stats)


def get_image_size(image_path: Path) -> tuple[int, int]:
    """
    Get image dimensions without loading full image into memory.
    Falls back to full cv2.imread if fast method fails.

    Returns:
        (width, height)
    """
    # Try fast header-only read first
    try:
        img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if img is not None:
            h, w = img.shape[:2]
            return w, h
    except Exception:
        pass

    raise IOError(f"Cannot read image: {image_path}")


def write_yolo_label(label_path: Path, class_ids: list[int], bboxes: list[list[float]]) -> None:
    """Write YOLOv8 format label file."""
    with open(label_path, "w") as f:
        for cls_id, bbox in zip(class_ids, bboxes):
            cx, cy, w, h = bbox
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


# ─────────────────────────────────────────────
#  BATCH CONVERTER
# ─────────────────────────────────────────────

def convert_dota_split(
    images_dir: Path,
    labels_dir: Path,
    output_images_dir: Path,
    output_labels_dir: Path,
    skip_difficult: bool = False,
    copy_images: bool = True,
    dry_run: bool = False,
    min_bbox_size: int = 4,
) -> dict:
    """
    Convert an entire DOTA dataset split to YOLOv8 format.

    Args:
        images_dir:         DOTA images folder (e.g. data/raw/dota/images/train)
        labels_dir:         DOTA labelTxt folder (e.g. data/raw/dota/labelTxt/train)
        output_images_dir:  destination for copied images
        output_labels_dir:  destination for converted YOLO labels
        skip_difficult:     skip annotations with difficulty=2
        copy_images:        if True, copy images to output_images_dir
        dry_run:            if True, process but don't write any files
        min_bbox_size:      minimum bbox dimension in pixels
    Returns:
        aggregated stats dict
    """
    if not dry_run:
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)

    # Find all images
    image_extensions = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    image_paths = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ])

    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")

    total_stats = defaultdict(int)
    total_stats["images_found"] = len(image_paths)

    for img_path in tqdm(image_paths, desc=f"Converting DOTA ({images_dir.name})"):

        # Find matching label file
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            total_stats["images_no_label"] += 1
            continue

        # Get image dimensions
        try:
            img_w, img_h = get_image_size(img_path)
        except IOError:
            total_stats["images_unreadable"] += 1
            continue

        # Convert annotations
        class_ids, bboxes, file_stats = convert_dota_annotation(
            label_path, img_w, img_h,
            skip_difficult=skip_difficult,
            min_bbox_size=min_bbox_size,
        )

        # Accumulate stats
        for k, v in file_stats.items():
            total_stats[k] += v

        if not class_ids:
            # No valid annotations after filtering — skip image
            total_stats["images_no_valid_annotations"] += 1
            continue

        total_stats["images_converted"] += 1

        if not dry_run:
            # Copy image
            if copy_images:
                dst_img = output_images_dir / img_path.name
                shutil.copy2(img_path, dst_img)

            # Write YOLO label
            dst_lbl = output_labels_dir / (img_path.stem + ".txt")
            write_yolo_label(dst_lbl, class_ids, bboxes)

    return dict(total_stats)


# ─────────────────────────────────────────────
#  VERIFICATION HELPERS
# ─────────────────────────────────────────────

def verify_conversion(
    output_images_dir: Path,
    output_labels_dir: Path,
    sample_size: int = 5,
) -> None:
    """
    Quick sanity check on converted dataset.
    Prints stats and verifies label format.

    Args:
        output_images_dir: converted images folder
        output_labels_dir: converted labels folder
        sample_size:       number of labels to spot-check
    """
    label_files = list(output_labels_dir.glob("*.txt"))
    image_files = list(output_images_dir.glob("*.jpg")) + list(output_images_dir.glob("*.png"))

    print(f"\n── Verification ─────────────────────────────")
    print(f"  Images : {len(image_files)}")
    print(f"  Labels : {len(label_files)}")

    class_counts = defaultdict(int)
    malformed = 0
    empty = 0

    for lbl in label_files:
        lines = lbl.read_text().strip().splitlines()
        if not lines:
            empty += 1
            continue
        for line in lines:
            parts = line.split()
            if len(parts) != 5:
                malformed += 1
                continue
            try:
                cls_id = int(parts[0])
                vals = [float(x) for x in parts[1:]]
                if cls_id in SV_CLASSES:
                    class_counts[SV_CLASSES[cls_id]] += 1
                # Check normalization
                if any(v < 0 or v > 1 for v in vals):
                    malformed += 1
            except ValueError:
                malformed += 1

    print(f"  Empty labels   : {empty}")
    print(f"  Malformed lines: {malformed}")
    print(f"\n  Class distribution:")
    for cls_name, count in sorted(class_counts.items()):
        print(f"    {cls_name:<15} {count:>6} instances")

    # Spot-check a few labels
    if label_files and sample_size > 0:
        print(f"\n  Sample labels (first {min(sample_size, len(label_files))}):")
        for lbl in label_files[:sample_size]:
            lines = lbl.read_text().strip().splitlines()
            print(f"    {lbl.name}: {len(lines)} annotations")

    print("─────────────────────────────────────────────\n")


# ─────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DOTA v2 annotations to YOLOv8 format for Sovereign-Vision",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument("--images",         required=True, help="DOTA images directory")
    parser.add_argument("--labels",         required=True, help="DOTA labelTxt directory")
    parser.add_argument("--output-images",  required=True, help="Output images directory")
    parser.add_argument("--output-labels",  required=True, help="Output labels directory")
    parser.add_argument("--split",          default="train",
                        choices=["train", "val", "test"],
                        help="Dataset split name (used in logs only, default: train)")
    parser.add_argument("--skip-difficult", action="store_true",
                        help="Skip annotations marked as difficult (difficulty=2)")
    parser.add_argument("--no-copy-images", action="store_true",
                        help="Don't copy images to output dir (labels only)")
    parser.add_argument("--min-bbox",       type=int, default=4,
                        help="Minimum bounding box size in pixels (default: 4)")
    parser.add_argument("--dry-run",        action="store_true",
                        help="Process without writing any files (stats only)")
    parser.add_argument("--verify",         action="store_true",
                        help="Run verification check after conversion")
    return parser.parse_args()


def print_stats(stats: dict, split: str) -> None:
    print(f"\n── DOTA → YOLOv8 Conversion: {split} ─────────────────")
    print(f"  Images found              : {stats.get('images_found', 0)}")
    print(f"  Images converted          : {stats.get('images_converted', 0)}")
    print(f"  Images (no label file)    : {stats.get('images_no_label', 0)}")
    print(f"  Images (no valid annot.)  : {stats.get('images_no_valid_annotations', 0)}")
    print(f"  Images unreadable         : {stats.get('images_unreadable', 0)}")
    print(f"\n  Annotations kept:")
    for sv_name in SV_CLASSES.values():
        count = stats.get(f"kept_{sv_name}", 0)
        print(f"    {sv_name:<15} {count:>6}")
    print(f"\n  Annotations skipped:")
    print(f"    difficult              : {stats.get('skipped_difficult', 0)}")
    print(f"    ignored class          : {stats.get('skipped_ignored', 0)}")
    print(f"    unknown class          : {stats.get('skipped_unknown', 0)}")
    print(f"    too small (< {stats.get('min_bbox', 4)}px)    : {stats.get('skipped_tiny', 0)}")
    print(f"    out of bounds          : {stats.get('skipped_oob', 0)}")
    print("──────────────────────────────────────────────────────\n")


def main():
    args = parse_args()

    images_dir        = Path(args.images)
    labels_dir        = Path(args.labels)
    output_images_dir = Path(args.output_images)
    output_labels_dir = Path(args.output_labels)

    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    if args.dry_run:
        print("── DRY RUN — no files will be written ──")

    stats = convert_dota_split(
        images_dir=images_dir,
        labels_dir=labels_dir,
        output_images_dir=output_images_dir,
        output_labels_dir=output_labels_dir,
        skip_difficult=args.skip_difficult,
        copy_images=not args.no_copy_images,
        dry_run=args.dry_run,
        min_bbox_size=args.min_bbox,
    )

    stats["min_bbox"] = args.min_bbox
    print_stats(stats, args.split)

    if args.verify and not args.dry_run:
        verify_conversion(output_images_dir, output_labels_dir)


if __name__ == "__main__":
    main()