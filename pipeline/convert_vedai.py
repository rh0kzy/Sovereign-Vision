import os
import math
import shutil
import argparse
import numpy as np
import cv2
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

SV_CLASSES = {0: "vehicle", 1: "truck", 2: "personnel", 3: "equipment"}

VEDAI_TO_SV = {
    1: 0, 2: 0, 9: 0, 10: 0, 11: 0,
    3: 1, 4: 1, 5: 1,
    6: 3, 7: 3, 8: 3,
}

VEDAI_CLASS_NAMES = {
    1: "car", 2: "pickup", 3: "truck", 4: "large-truck",
    5: "tractor", 6: "plane", 7: "boat", 8: "motorcycle",
    9: "camping-car", 10: "other", 11: "van",
}


def rotated_bbox_to_aabb(cx, cy, w, h, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    half_w, half_h = w / 2, h / 2
    corners_local = [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
    corners = [(cx + dx * cos_a - dy * sin_a, cy + dx * sin_a + dy * cos_a) for dx, dy in corners_local]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return min(xs), min(ys), max(xs), max(ys)


def aabb_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h):
    cx = (x_min + x_max) / 2 / img_w
    cy = (y_min + y_max) / 2 / img_h
    w  = (x_max - x_min) / img_w
    h  = (y_max - y_min) / img_h
    return (max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy)),
            max(0.0, min(1.0, w)),  max(0.0, min(1.0, h)))


def parse_vedai_annotation(label_path):
    annotations = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            try:
                annotations.append({
                    "cx": float(parts[1]), "cy": float(parts[2]),
                    "angle": float(parts[3]), "w": float(parts[4]), "h": float(parts[5]),
                    "occluded": int(parts[6]), "cut": int(parts[7]), "class_id": int(parts[8]),
                })
            except (ValueError, IndexError):
                continue
    return annotations


def convert_vedai_annotation(label_path, img_w, img_h, skip_occluded=False, skip_cut=False, min_bbox_size=4):
    annotations = parse_vedai_annotation(label_path)
    class_ids, bboxes, stats = [], [], defaultdict(int)
    for ann in annotations:
        if skip_occluded and ann["occluded"] == 1:
            stats["skipped_occluded"] += 1
            continue
        if skip_cut and ann["cut"] == 1:
            stats["skipped_cut"] += 1
            continue
        vedai_cls = ann["class_id"]
        if vedai_cls not in VEDAI_TO_SV:
            stats["skipped_unknown"] += 1
            continue
        sv_class_id = VEDAI_TO_SV[vedai_cls]
        x_min, y_min, x_max, y_max = rotated_bbox_to_aabb(
            ann["cx"], ann["cy"], ann["w"], ann["h"], ann["angle"])
        x_min = max(0.0, x_min); y_min = max(0.0, y_min)
        x_max = min(float(img_w), x_max); y_max = min(float(img_h), y_max)
        if (x_max - x_min) < min_bbox_size or (y_max - y_min) < min_bbox_size:
            stats["skipped_tiny"] += 1
            continue
        cx_n, cy_n, w_n, h_n = aabb_to_yolo(x_min, y_min, x_max, y_max, img_w, img_h)
        class_ids.append(sv_class_id)
        bboxes.append([cx_n, cy_n, w_n, h_n])
        stats[f"kept_{SV_CLASSES[sv_class_id]}"] += 1
        stats[f"src_{VEDAI_CLASS_NAMES.get(vedai_cls, str(vedai_cls))}"] += 1
    return class_ids, bboxes, dict(stats)


def get_image_size(image_path):
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"Cannot read image: {image_path}")
    h, w = img.shape[:2]
    return w, h


def write_yolo_label(label_path, class_ids, bboxes):
    with open(label_path, "w") as f:
        for cls_id, bbox in zip(class_ids, bboxes):
            cx, cy, w, h = bbox
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def find_image_for_annotation(label_stem, images_dir):
    for candidate in [
        images_dir / f"co_{label_stem}.png",
        images_dir / f"ir_{label_stem}.png",
        images_dir / f"{label_stem}.png",
        images_dir / f"{label_stem}.jpg",
    ]:
        if candidate.exists():
            return candidate
    return None


def convert_vedai_dataset(images_dir, labels_dir, output_images_dir, output_labels_dir,
                          skip_occluded=False, skip_cut=False, copy_images=True,
                          dry_run=False, min_bbox_size=4):
    if not dry_run:
        output_images_dir.mkdir(parents=True, exist_ok=True)
        output_labels_dir.mkdir(parents=True, exist_ok=True)
    label_files = sorted(labels_dir.glob("*.txt"))
    if not label_files:
        raise FileNotFoundError(f"No annotation files found in {labels_dir}")
    total_stats = defaultdict(int)
    total_stats["labels_found"] = len(label_files)
    for lbl_path in tqdm(label_files, desc="Converting VEDAI"):
        img_path = find_image_for_annotation(lbl_path.stem, images_dir)
        if img_path is None:
            total_stats["images_not_found"] += 1
            continue
        try:
            img_w, img_h = get_image_size(img_path)
        except IOError:
            total_stats["images_unreadable"] += 1
            continue
        class_ids, bboxes, file_stats = convert_vedai_annotation(
            lbl_path, img_w, img_h, skip_occluded=skip_occluded,
            skip_cut=skip_cut, min_bbox_size=min_bbox_size)
        for k, v in file_stats.items():
            total_stats[k] += v
        if not class_ids:
            total_stats["images_no_valid_annotations"] += 1
            continue
        total_stats["images_converted"] += 1
        if not dry_run:
            if copy_images:
                shutil.copy2(img_path, output_images_dir / img_path.name)
            write_yolo_label(output_labels_dir / (img_path.stem + ".txt"), class_ids, bboxes)
    return dict(total_stats)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert VEDAI to YOLOv8 for Sovereign-Vision")
    parser.add_argument("--images",         required=True)
    parser.add_argument("--labels",         required=True)
    parser.add_argument("--output-images",  required=True)
    parser.add_argument("--output-labels",  required=True)
    parser.add_argument("--skip-occluded",  action="store_true")
    parser.add_argument("--skip-cut",       action="store_true")
    parser.add_argument("--no-copy-images", action="store_true")
    parser.add_argument("--min-bbox",       type=int, default=4)
    parser.add_argument("--dry-run",        action="store_true")
    parser.add_argument("--verify",         action="store_true")
    return parser.parse_args()


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
    stats = convert_vedai_dataset(
        images_dir=images_dir, labels_dir=labels_dir,
        output_images_dir=output_images_dir, output_labels_dir=output_labels_dir,
        skip_occluded=args.skip_occluded, skip_cut=args.skip_cut,
        copy_images=not args.no_copy_images, dry_run=args.dry_run,
        min_bbox_size=args.min_bbox)
    print(f"\n── VEDAI → YOLOv8 Conversion ─────────────────")
    print(f"  Labels found              : {stats.get('labels_found', 0)}")
    print(f"  Images converted          : {stats.get('images_converted', 0)}")
    print(f"  Images not found          : {stats.get('images_not_found', 0)}")
    print(f"  Images (no valid annot.)  : {stats.get('images_no_valid_annotations', 0)}")
    print(f"\n  Annotations kept:")
    for sv_name in SV_CLASSES.values():
        print(f"    {sv_name:<15} {stats.get(f'kept_{sv_name}', 0):>6}")
    print("──────────────────────────────────────────────\n")
    if args.verify and not args.dry_run:
        print("✓ Conversion complete.")


if __name__ == "__main__":
    main()