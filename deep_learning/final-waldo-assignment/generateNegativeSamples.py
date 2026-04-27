"""
Generate negative samples and combine with true samples into /kaggle/working.
Run this in a Kaggle notebook cell after the label-generation script.
"""
import os
import shutil
import random
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm

random.seed(123)
np.random.seed(123)

# ── Kaggle paths ──────────────────────────────────────────────────────────────
BASE        = Path("/kaggle/input/datasets/sheshngupta/waldowilma/generated_images")
RESIZED_DIR = BASE / "resized_224x224"

OUT         = Path("/kaggle/working")
LABELS_DIR  = OUT / "labels_resized_224x224"   # already created by label-gen script

COMBINED_IMGS   = OUT / "images_combined"
COMBINED_LABELS = OUT / "labels_combined"
COMBINED_IMGS.mkdir(exist_ok=True)
COMBINED_LABELS.mkdir(exist_ok=True)

N_NEGATIVE = 500

# ── Step 1: compute median background from input images ───────────────────────
print("Computing median background...")
all_img_files = sorted(RESIZED_DIR.glob("*.png"))
sample = random.sample(all_img_files, 50)
stack  = np.stack([
    np.array(Image.open(p).convert("RGB"), dtype=np.uint8)
    for p in sample
], axis=0)
background = np.median(stack, axis=0).astype(np.uint8)
print(f"  Background shape: {background.shape}")

# ── Step 2: generate negative images + empty labels ──────────────────────────
print(f"\nGenerating {N_NEGATIVE} negative samples...")
for i in tqdm(range(N_NEGATIVE)):
    img = background.astype(np.float32).copy()

    sigma      = np.random.uniform(5, 15)
    img       += np.random.normal(0, sigma, img.shape)
    img       += np.random.uniform(-20, 20)          # brightness
    mean       = img.mean()
    img        = (img - mean) * np.random.uniform(0.85, 1.15) + mean  # contrast
    img        = np.clip(img, 0, 255).astype(np.uint8)

    name = f"negative_{i:04d}"
    Image.fromarray(img).save(COMBINED_IMGS / f"{name}.png")
    open(COMBINED_LABELS / f"{name}.txt", "w").close()   # empty = no objects

# ── Step 3: copy true samples (images + labels) into combined folders ─────────
print("\nCopying true samples into combined folders...")

true_imgs = sorted(RESIZED_DIR.glob("*.png"))
missing_labels = 0

for img_path in tqdm(true_imgs):
    label_path = LABELS_DIR / (img_path.stem + ".txt")

    shutil.copy(img_path,   COMBINED_IMGS   / img_path.name)

    if label_path.exists():
        shutil.copy(label_path, COMBINED_LABELS / label_path.name)
    else:
        # safety fallback: treat as negative rather than silently skip
        open(COMBINED_LABELS / (img_path.stem + ".txt"), "w").close()
        missing_labels += 1

if missing_labels:
    print(f"  ⚠  {missing_labels} images had no matching label — written as negatives")

# ── Step 4: sanity check ──────────────────────────────────────────────────────
n_imgs   = len(list(COMBINED_IMGS.glob("*.png")))
n_labels = len(list(COMBINED_LABELS.glob("*.txt")))
n_empty  = sum(
    1 for p in COMBINED_LABELS.glob("*.txt") if p.stat().st_size == 0
)

print(f"\n{'─'*50}")
print(f"Combined dataset written to {OUT}")
print(f"  Images : {n_imgs:>6}")
print(f"  Labels : {n_labels:>6}  ({n_empty} empty / negatives)")
print(f"  Mismatch: {abs(n_imgs - n_labels)}")
