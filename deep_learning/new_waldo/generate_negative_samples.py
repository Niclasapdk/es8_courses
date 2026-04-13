"""
Generate negative samples (background-only images) for YOLO training.

Computes the median background from existing images, then creates 500
variations with random noise, brightness, and contrast jitter so the
model learns to suppress detections when no character is present.

Each negative sample gets an empty .txt label file (YOLO convention
for "no objects in this image").
"""

import os
import random
import numpy as np
from PIL import Image

random.seed(123)
np.random.seed(123)

HERE    = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "resized_224x224")
OUT_IMG = os.path.join(HERE, "resized_224x224")          # same folder as real images
OUT_LBL = os.path.join(HERE, "labels_resized_224x224")   # same folder as real labels

N_NEGATIVE = 500   # ~10% of 5000

# ── Step 1: compute median background ────────────────────────────────────────

print("Computing median background from 50 random images...")
all_imgs = sorted([f for f in os.listdir(IMG_DIR) if f.startswith("resized_image_")])
sample   = random.sample(all_imgs, 50)
stack    = np.stack([
    np.array(Image.open(os.path.join(IMG_DIR, f)).convert("RGB"), dtype=np.uint8)
    for f in sample
], axis=0)
background = np.median(stack, axis=0).astype(np.uint8)
print(f"  Background shape: {background.shape}")

# ── Step 2: generate variations ──────────────────────────────────────────────

print(f"Generating {N_NEGATIVE} negative samples...")
for i in range(N_NEGATIVE):
    img = background.astype(np.float32).copy()

    # Random Gaussian noise (sigma 5–15)
    sigma = np.random.uniform(5, 15)
    noise = np.random.normal(0, sigma, img.shape)
    img += noise

    # Random brightness shift (-20 to +20)
    brightness = np.random.uniform(-20, 20)
    img += brightness

    # Random contrast (0.85 to 1.15)
    contrast = np.random.uniform(0.85, 1.15)
    mean = img.mean()
    img = (img - mean) * contrast + mean

    # Clip and convert
    img = np.clip(img, 0, 255).astype(np.uint8)

    name = f"negative_{i}"
    Image.fromarray(img).save(os.path.join(OUT_IMG, name + ".png"))

    # Empty label file = no objects
    open(os.path.join(OUT_LBL, name + ".txt"), "w").close()

print(f"Done. Saved {N_NEGATIVE} images to {OUT_IMG}")
print(f"       {N_NEGATIVE} empty labels to {OUT_LBL}")
print(f"\nTotal dataset is now {len(all_imgs) + N_NEGATIVE} images (5000 + {N_NEGATIVE} negative)")
