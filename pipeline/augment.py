"""
sovereign-vision / pipeline / augment.py
=========================================
Desert-environment augmentation pipeline for aerial surveillance imagery.
Designed for North African arid terrain (Saharan conditions).

Augmentation modes
------------------
1. sand_tone       – HSV shift toward ochre/beige desert spectrum
2. heat_shimmer    – sinusoidal warp + Gaussian blur simulating thermal air refraction
3. low_res_sim     – bicubic downscale → upscale to mimic cheap drone optics
4. glare_injection – random bright elliptical spots simulating desert sun lens flare
5. dust_overlay    – semi-transparent noise layer for sandstorm / haze conditions
6. thermal_sim     – convert RGB frame to FLIR-style thermal colormap (INFERNO / HOT)

Usage
-----
python augment.py --input data/raw/sample.jpg --output data/augmented/ --compare
python augment.py --input data/processed/images/ --labels data/processed/labels/ \
                  --output data/augmented/ --copies 5 --modes sand,shimmer,lowres,glare
python augment.py --input data/processed/images/ --output data/augmented/thermal/ \
                  --modes thermal --colormap inferno

Requirements
------------
pip install opencv-python albumentations numpy tqdm
"""

import cv2
import numpy as np
import albumentations as A
import argparse
import os
import random
from pathlib import Path
from tqdm import tqdm


SAND_HUE_SHIFT   = (10, 25)
SAND_SAT_REDUCE  = (0.3, 0.6)
SAND_VAL_BOOST   = (1.05, 1.20)

THERMAL_COLORMAPS = {
    "inferno": cv2.COLORMAP_INFERNO,
    "hot":     cv2.COLORMAP_HOT,
    "jet":     cv2.COLORMAP_JET,
    "bone":    cv2.COLORMAP_BONE,
}


def sand_tone(image: np.ndarray, intensity: float = 0.7) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    h_shift = random.uniform(*SAND_HUE_SHIFT) * intensity
    s_factor = random.uniform(*SAND_SAT_REDUCE) * intensity + (1 - intensity)
    v_factor = random.uniform(*SAND_VAL_BOOST)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + h_shift, 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * s_factor, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * v_factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def heat_shimmer(image: np.ndarray, strength: float = 5.0, frequency: float = 0.03) -> np.ndarray:
    h, w = image.shape[:2]
    x_coords = np.tile(np.arange(w), (h, 1)).astype(np.float32)
    y_coords = np.tile(np.arange(h), (w, 1)).T.astype(np.float32)
    phase_x = random.uniform(0, 2 * np.pi)
    phase_y = random.uniform(0, 2 * np.pi)
    x_disp = strength * np.sin(2 * np.pi * frequency * y_coords + phase_x)
    y_disp = strength * np.sin(2 * np.pi * frequency * x_coords + phase_y)
    map_x = np.clip(x_coords + x_disp, 0, w - 1).astype(np.float32)
    map_y = np.clip(y_coords + y_disp, 0, h - 1).astype(np.float32)
    warped = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                       borderMode=cv2.BORDER_REFLECT)
    blur_k = random.choice([3, 5])
    return cv2.GaussianBlur(warped, (blur_k, blur_k), 0)


def low_res_sim(image: np.ndarray, scale_factor: float = None) -> np.ndarray:
    h, w = image.shape[:2]
    if scale_factor is None:
        scale_factor = random.uniform(0.25, 0.55)
    small_w = max(int(w * scale_factor), 32)
    small_h = max(int(h * scale_factor), 32)
    small = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def glare_injection(image: np.ndarray, n_spots: int = None) -> np.ndarray:
    result = image.copy().astype(np.float32)
    h, w = image.shape[:2]
    if n_spots is None:
        n_spots = random.randint(1, 3)
    for _ in range(n_spots):
        cx = random.randint(0, w)
        cy = random.randint(0, h)
        rx = random.randint(w // 20, w // 6)
        ry = random.randint(h // 20, h // 6)
        angle = random.uniform(0, 180)
        intensity = random.uniform(0.4, 0.85)
        mask = np.zeros((h, w), dtype=np.float32)
        cv2.ellipse(mask, (cx, cy), (rx, ry), angle, 0, 360, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        glare_color = np.array([220, 240, 255], dtype=np.float32)
        result += mask[:, :, np.newaxis] * glare_color * intensity
    return np.clip(result, 0, 255).astype(np.uint8)


def dust_overlay(image: np.ndarray, density: float = None) -> np.ndarray:
    h, w = image.shape[:2]
    if density is None:
        density = random.uniform(0.10, 0.45)
    noise = np.random.randint(180, 230, (h, w), dtype=np.uint8)
    noise_blurred = cv2.GaussianBlur(noise, (0, 0), sigmaX=random.uniform(20, 60))
    dust_color = np.stack([
        (noise_blurred * 0.75).astype(np.uint8),
        (noise_blurred * 0.88).astype(np.uint8),
        noise_blurred
    ], axis=-1)
    return cv2.addWeighted(image, 1 - density, dust_color, density, 0)


def thermal_sim(image: np.ndarray, colormap: str = "inferno", noise_level: float = 8.0) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, equalized.shape).astype(np.float32)
        equalized = np.clip(equalized.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    cmap = THERMAL_COLORMAPS.get(colormap, cv2.COLORMAP_INFERNO)
    return cv2.applyColorMap(equalized, cmap)


def build_albumentations_pipeline(p: float = 0.8) -> A.Compose:
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=20,
                           border_mode=cv2.BORDER_REFLECT, p=0.5),
        A.Perspective(scale=(0.02, 0.08), p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.6),
        A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.Sharpen(alpha=(0.2, 0.5), p=0.3),
        A.GaussNoise(var_limit=(10, 50), p=0.3),
        A.MotionBlur(blur_limit=5, p=0.2),
        A.ImageCompression(quality_lower=60, quality_upper=95, p=0.4),
        A.CoarseDropout(num_holes_range=(1, 4), hole_height_range=(0.02, 0.08),
                        hole_width_range=(0.02, 0.08), p=0.2),
    ],
    bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.3))


def read_yolo_labels(label_path: Path) -> tuple:
    class_ids, bboxes = [], []
    if not label_path.exists():
        return class_ids, bboxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                cls = int(parts[0])
                bbox = [max(0.0, min(1.0, float(x))) for x in parts[1:]]
                class_ids.append(cls)
                bboxes.append(bbox)
    return class_ids, bboxes


def write_yolo_labels(label_path: Path, class_ids: list, bboxes: list) -> None:
    with open(label_path, "w") as f:
        for cls, bbox in zip(class_ids, bboxes):
            cx, cy, w, h = bbox
            f.write(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


DESERT_MODES = {
    "sand":    lambda img: sand_tone(img),
    "shimmer": lambda img: heat_shimmer(img),
    "lowres":  lambda img: low_res_sim(img),
    "glare":   lambda img: glare_injection(img),
    "dust":    lambda img: dust_overlay(img),
    "thermal": lambda img: thermal_sim(img),
}

DEFAULT_MODES = ["sand", "shimmer", "lowres", "glare", "dust"]


def augment_image_and_labels(image, class_ids, bboxes, modes=None, albu_pipeline=None, apply_desert=True):
    if modes is None:
        modes = DEFAULT_MODES
    if albu_pipeline is None:
        albu_pipeline = build_albumentations_pipeline()
    if class_ids and bboxes:
        result = albu_pipeline(image=image, bboxes=bboxes, class_labels=class_ids)
        image = result["image"]
        bboxes = list(result["bboxes"])
        class_ids = list(result["class_labels"])
    else:
        aug = A.Compose([t for t in albu_pipeline.transforms])
        image = aug(image=image)["image"]
    if apply_desert:
        active_modes = [m for m in modes if random.random() < 0.65]
        for mode in active_modes:
            if mode in DESERT_MODES:
                image = DESERT_MODES[mode](image)
    return image, class_ids, bboxes


def augment_dataset(images_dir, labels_dir, output_images_dir, output_labels_dir,
                    copies=5, modes=None, thermal_copy=True):
    output_images_dir.mkdir(parents=True, exist_ok=True)
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted([p for p in images_dir.iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])
    if not image_paths:
        raise FileNotFoundError(f"No images found in {images_dir}")
    albu_pipeline = build_albumentations_pipeline()
    stats = {"source": 0, "augmented": 0, "thermal": 0, "skipped": 0}
    for img_path in tqdm(image_paths, desc="Augmenting dataset"):
        image = cv2.imread(str(img_path))
        if image is None:
            stats["skipped"] += 1
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        class_ids, bboxes = read_yolo_labels(label_path)
        stats["source"] += 1
        dst_img = output_images_dir / img_path.name
        dst_lbl = output_labels_dir / (img_path.stem + ".txt")
        cv2.imwrite(str(dst_img), image)
        if class_ids:
            write_yolo_labels(dst_lbl, class_ids, bboxes)
        for i in range(copies):
            aug_img, aug_cls, aug_bboxes = augment_image_and_labels(
                image.copy(), list(class_ids), list(bboxes),
                modes=modes, albu_pipeline=albu_pipeline)
            suffix = f"_aug{i:02d}"
            out_img_path = output_images_dir / f"{img_path.stem}{suffix}{img_path.suffix}"
            out_lbl_path = output_labels_dir / f"{img_path.stem}{suffix}.txt"
            cv2.imwrite(str(out_img_path), aug_img)
            if aug_cls:
                write_yolo_labels(out_lbl_path, aug_cls, aug_bboxes)
            stats["augmented"] += 1
        if thermal_copy:
            thermal_img = thermal_sim(image)
            t_img_path = output_images_dir / f"{img_path.stem}_thermal{img_path.suffix}"
            t_lbl_path = output_labels_dir / f"{img_path.stem}_thermal.txt"
            cv2.imwrite(str(t_img_path), thermal_img)
            if class_ids:
                write_yolo_labels(t_lbl_path, class_ids, bboxes)
            stats["thermal"] += 1
    return stats


def make_comparison_grid(image, size=320):
    def resize(img):
        return cv2.resize(img, (size, size))
    cells = [("Original", resize(image))]
    for name, fn in DESERT_MODES.items():
        try:
            aug = fn(image.copy())
            cells.append((name.capitalize(), resize(aug)))
        except Exception:
            blank = np.zeros((size, size, 3), dtype=np.uint8)
            cells.append((name, blank))
    n = len(cells)
    cols = 4
    rows = (n + cols - 1) // cols
    row_imgs = []
    for r in range(rows):
        row_cells = []
        for c in range(cols):
            idx = r * cols + c
            if idx < n:
                label, cell = cells[idx]
                labeled = np.zeros((size + 28, size, 3), dtype=np.uint8)
                labeled[28:] = cell
                cv2.putText(labeled, label, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
            else:
                labeled = np.zeros((size + 28, size, 3), dtype=np.uint8)
            row_cells.append(labeled)
        row_imgs.append(np.hstack(row_cells))
    return np.vstack(row_imgs)


def parse_args():
    parser = argparse.ArgumentParser(description="Sovereign-Vision Desert Augmentation Pipeline")
    parser.add_argument("--input",    required=True)
    parser.add_argument("--labels",   default=None)
    parser.add_argument("--output",   required=True)
    parser.add_argument("--copies",   type=int, default=5)
    parser.add_argument("--modes",    default=",".join(DEFAULT_MODES))
    parser.add_argument("--colormap", default="inferno", choices=list(THERMAL_COLORMAPS.keys()))
    parser.add_argument("--compare",  action="store_true")
    parser.add_argument("--no-thermal", action="store_true")
    parser.add_argument("--seed",     type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    modes = [m.strip() for m in args.modes.split(",") if m.strip() in DESERT_MODES]
    if not modes:
        raise ValueError(f"No valid modes. Choose from: {list(DESERT_MODES.keys())}")
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    if input_path.is_file():
        image = cv2.imread(str(input_path))
        if image is None:
            raise IOError(f"Cannot read image: {input_path}")
        if args.compare:
            grid = make_comparison_grid(image)
            grid_path = output_path / f"{input_path.stem}_comparison.jpg"
            cv2.imwrite(str(grid_path), grid)
            print(f"✓ Comparison grid saved → {grid_path}")
        class_ids, bboxes = [], []
        if args.labels:
            lbl = Path(args.labels) / (input_path.stem + ".txt")
            class_ids, bboxes = read_yolo_labels(lbl)
        for i in range(args.copies):
            aug_img, aug_cls, aug_bboxes = augment_image_and_labels(
                image.copy(), list(class_ids), list(bboxes), modes=modes)
            out = output_path / f"{input_path.stem}_aug{i:02d}{input_path.suffix}"
            cv2.imwrite(str(out), aug_img)
            if aug_cls:
                write_yolo_labels(output_path / f"{input_path.stem}_aug{i:02d}.txt", aug_cls, aug_bboxes)
        if not args.no_thermal:
            t_img = thermal_sim(image, colormap=args.colormap)
            cv2.imwrite(str(output_path / f"{input_path.stem}_thermal{input_path.suffix}"), t_img)
        print(f"✓ Generated {args.copies} augmented copies + thermal → {output_path}")
    elif input_path.is_dir():
        labels_dir = Path(args.labels) if args.labels else input_path.parent / "labels"
        out_imgs = output_path / "images"
        out_lbls = output_path / "labels"
        stats = augment_dataset(input_path, labels_dir, out_imgs, out_lbls,
                                copies=args.copies, modes=modes, thermal_copy=not args.no_thermal)
        print("\n── Augmentation Complete ──────────────────")
        print(f"  Source images   : {stats['source']}")
        print(f"  Augmented copies: {stats['augmented']}")
        print(f"  Thermal copies  : {stats['thermal']}")
        print(f"  Skipped         : {stats['skipped']}")
        print(f"  Total output    : {stats['source'] + stats['augmented'] + stats['thermal']}")
        print(f"  Output directory: {output_path}")
        print("──────────────────────────────────────────\n")
    else:
        raise FileNotFoundError(f"Input not found: {input_path}")


if __name__ == "__main__":
    main()