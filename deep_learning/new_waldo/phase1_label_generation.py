"""
Phase 1 - Label Generation via Image Differencing
==================================================
Since all images share the same background (synthetically generated),
we compute a median background from a sample of images and subtract it
from each image. The remaining non-zero region is where the character
was pasted. We then determine which character (waldo/wilma) it is by
comparing the cropped region against both templates.

Outputs (written to /kaggle/working):
  labels_original_350x500/   YOLO .txt labels
  labels_resized_224x224/    YOLO .txt labels
  annotations_original.csv   human-readable summary
  annotations_resized.csv    human-readable summary

YOLO format:  <class> <cx_norm> <cy_norm> <w_norm> <h_norm>
  class 0 = waldo,  class 1 = wilma
"""

import cv2
import numpy as np
from pathlib import Path
import csv
import random
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE     = Path("/kaggle/input/datasets/sheshngupta/waldowilma/generated_images")
OUT_BASE = Path("/kaggle/working")

CHAR_DIR    = BASE / "charecters"
ORIG_DIR    = BASE / "original_350x500"
RESIZED_DIR = BASE / "resized_224x224"

LABELS_ORIG    = OUT_BASE / "labels_original_350x500"
LABELS_RESIZED = OUT_BASE / "labels_resized_224x224"
LABELS_ORIG.mkdir(exist_ok=True)
LABELS_RESIZED.mkdir(exist_ok=True)

CLASSES = {"waldo": 0, "wilma": 1}

# ── Load image as numpy RGB array ──────────────────────────────────────────────
def load_rgb(path: Path) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(str(path)).convert("RGB"), dtype=np.uint8)


# ── Step 1: Compute median background ─────────────────────────────────────────
def compute_background(image_dir: Path, n_samples: int = 50) -> np.ndarray:
    """
    Sample n images and take the per-pixel median.
    Since each image has a character pasted in a different location,
    the median across enough images converges to the clean background.
    """
    paths = random.sample(sorted(image_dir.glob("*.png")), n_samples)
    stack = np.stack([load_rgb(p) for p in paths], axis=0)  # (N, H, W, 3)
    background = np.median(stack, axis=0).astype(np.uint8)
    return background


# ── Step 2: Find character bounding box via diff ───────────────────────────────
def find_character_bbox(img_rgb: np.ndarray, background: np.ndarray,
                        diff_threshold: int = 30) -> tuple:
    """
    Subtract background, threshold, find bounding box of changed pixels.
    Returns (x, y, w, h) in pixel coords, or None if nothing found.
    """
    diff = np.abs(img_rgb.astype(np.int16) - background.astype(np.int16))
    diff_gray = diff.max(axis=2).astype(np.uint8)   # per-pixel max channel diff
    _, mask = cv2.threshold(diff_gray, diff_threshold, 255, cv2.THRESH_BINARY)

    # Clean up noise with morphological ops
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask   = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Take the largest contour (the character)
    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    return x, y, w, h


# ── Step 3: Identify which character via masked MSE ───────────────────────────
def identify_character(crop_rgb: np.ndarray, templates: dict) -> str:
    """
    Resize each template to match the crop bounding box, apply its alpha mask,
    and compute MSE only over the non-transparent pixels.
    Lower MSE = better match. Returns 'waldo' or 'wilma'.
    """
    h, w = crop_rgb.shape[:2]

    best_char = None
    best_mse  = float("inf")

    for char_name, (tmpl_gray, tmpl_alpha) in templates.items():
        # Resize both the RGB template and its alpha mask to crop size
        tmpl_bgr   = cv2.cvtColor(tmpl_gray, cv2.COLOR_GRAY2BGR)
        resized_rgb   = cv2.resize(tmpl_bgr,   (w, h), interpolation=cv2.INTER_AREA)
        resized_alpha = cv2.resize(tmpl_alpha, (w, h), interpolation=cv2.INTER_AREA)

        # Only compare pixels where template is opaque (alpha > 128)
        mask = resized_alpha > 128
        if mask.sum() == 0:
            continue

        diff = crop_rgb.astype(np.float32) - resized_rgb.astype(np.float32)
        mse  = (diff[mask] ** 2).mean()

        if mse < best_mse:
            best_mse  = mse
            best_char = char_name

    return best_char


# ── Worker (multiprocessing) ───────────────────────────────────────────────────
_BACKGROUND = None
_TEMPLATES  = {}

def _init_worker(background, templates):
    global _BACKGROUND, _TEMPLATES
    _BACKGROUND = background
    _TEMPLATES  = templates


def _process_one(args):
    img_path_str, label_dir_str, diff_threshold = args
    img_path  = Path(img_path_str)
    label_dir = Path(label_dir_str)

    img_rgb = load_rgb(img_path)
    img_h, img_w = img_rgb.shape[:2]

    bbox = find_character_bbox(img_rgb, _BACKGROUND, diff_threshold)

    if bbox is None:
        # Fallback: mark centre with a small box and class -1 (unknown)
        x, y, w, h = img_w // 4, img_h // 4, img_w // 2, img_h // 2
        char_name  = "unknown"
        class_id   = -1
    else:
        x, y, w, h = bbox
        # Clamp
        x = max(0, x); y = max(0, y)
        w = min(w, img_w - x); h = min(h, img_h - y)

        crop    = img_rgb[y:y+h, x:x+w]
        char_name = identify_character(crop, _TEMPLATES)
        class_id  = CLASSES[char_name]

    cx_norm = (x + w / 2) / img_w
    cy_norm = (y + h / 2) / img_h
    w_norm  = w / img_w
    h_norm  = h / img_h

    label_file = label_dir / (img_path.stem + ".txt")
    label_file.write_text(
        f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}\n"
    )

    return (img_path.name, char_name, class_id,
            x, y, w, h, img_w, img_h,
            cx_norm, cy_norm, w_norm, h_norm)


# ── Per image-set driver ───────────────────────────────────────────────────────
def process_image_set(image_dir: Path, label_dir: Path, csv_path: Path,
                      background: np.ndarray, templates: dict,
                      diff_threshold: int, n_workers: int):

    image_paths = sorted(image_dir.glob("*.png"))
    args = [(str(p), str(label_dir), diff_threshold) for p in image_paths]

    rows = []
    with Pool(processes=n_workers,
              initializer=_init_worker,
              initargs=(background, templates)) as pool:
        for row in tqdm(pool.imap_unordered(_process_one, args, chunksize=50),
                        total=len(args), desc=f"{image_dir.name}"):
            rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "character", "class_id",
                         "x", "y", "w", "h", "img_w", "img_h",
                         "cx_norm", "cy_norm", "w_norm", "h_norm"])
        for r in rows:
            writer.writerow([r[0], r[1], r[2],
                             r[3], r[4], r[5], r[6], r[7], r[8],
                             f"{r[9]:.6f}", f"{r[10]:.6f}",
                             f"{r[11]:.6f}", f"{r[12]:.6f}"])

    n_unknown = sum(1 for r in rows if r[2] == -1)
    print(f"  {len(rows)} images processed, {n_unknown} with no detection")


# ── Template loading ───────────────────────────────────────────────────────────
def load_template_gray(path: Path):
    """Returns (gray, alpha) — gray is kept for compatibility but MSE uses BGR."""
    from PIL import Image
    img   = Image.open(str(path)).convert("RGBA")
    arr   = np.array(img)
    gray  = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    alpha = arr[:, :, 3]
    return gray, alpha


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    n_workers      = cpu_count()
    diff_threshold = 30   # pixel difference to count as "changed"

    print("Loading templates...")
    templates = {name: load_template_gray(CHAR_DIR / f"{name}.png")
                 for name in CLASSES}
    for name, (g, _) in templates.items():
        print(f"  {name}: {g.shape[1]}×{g.shape[0]}")

    print(f"\nWorkers: {n_workers}  |  Diff threshold: {diff_threshold}\n")

    # ── Original 350x500 set ──
    print("Computing background for original_350x500 (sampling 50 images)...")
    bg_orig = compute_background(ORIG_DIR, n_samples=50)
    print("  Done.")
    process_image_set(ORIG_DIR, LABELS_ORIG, OUT_BASE / "annotations_original.csv",
                      bg_orig, templates, diff_threshold, n_workers)

    # ── Resized 224x224 set ──
    print("\nComputing background for resized_224x224 (sampling 50 images)...")
    bg_resized = compute_background(RESIZED_DIR, n_samples=50)
    print("  Done.")
    process_image_set(RESIZED_DIR, LABELS_RESIZED, OUT_BASE / "annotations_resized.csv",
                      bg_resized, templates, diff_threshold, n_workers)

    print("\nDone.")
    print(f"  {LABELS_ORIG}")
    print(f"  {LABELS_RESIZED}")


if __name__ == "__main__":
    main()
