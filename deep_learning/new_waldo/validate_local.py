"""
Local validation script for the Waldo/Wilma YOLOv8 model.

Steps:
  1. Recreates the 70/15/15 train/val/test split (seed=42, same as Kaggle training)
     using symlinks — no files are copied.
  2. Writes dataset/ folder structure expected by YOLO.
  3. Runs model.val() on the test split to generate the confusion matrix.

Output will be saved to runs/detect/valN/ (auto-incremented by YOLO).
"""

import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_SRC = os.path.join(HERE, "resized_224x224")
LBL_SRC = os.path.join(HERE, "labels_resized_224x224")
DATASET = os.path.join(HERE, "dataset")
DATA_YAML = os.path.join(HERE, "data.yaml")
WEIGHTS = os.path.join(HERE, "best.pt")

# ── Step 1: build split ──────────────────────────────────────────────────────

random.seed(42)
names = sorted([f[:-4] for f in os.listdir(IMG_SRC) if f.endswith(".png")])
random.shuffle(names)

n = len(names)
splits = {
    "train": names[: int(0.70 * n)],
    "val":   names[int(0.70 * n) : int(0.85 * n)],
    "test":  names[int(0.85 * n) :],
}

# ── Step 2: create symlinked folder structure ────────────────────────────────

for split, items in splits.items():
    img_dir = os.path.join(DATASET, "images", split)
    lbl_dir = os.path.join(DATASET, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    for name in items:
        img_link = os.path.join(img_dir, name + ".png")
        lbl_link = os.path.join(lbl_dir, name + ".txt")

        if not os.path.exists(img_link):
            os.symlink(os.path.join(IMG_SRC, name + ".png"), img_link)
        if not os.path.exists(lbl_link):
            os.symlink(os.path.join(LBL_SRC, name + ".txt"), lbl_link)

    print(f"{split}: {len(items)} images")

# ── Step 3: validate ─────────────────────────────────────────────────────────

from ultralytics import YOLO

model = YOLO(WEIGHTS)
metrics = model.val(
    data=DATA_YAML,
    split="test",
    imgsz=224,
    batch=32,
    iou=0.3,       # stricter NMS — merge overlapping boxes more aggressively (default is 0.7)
    project=os.path.join(HERE, "runs/detect"),
    name="val",
    plots=True,   # generates confusion_matrix.png
)

print("\n── Results ──")
print(f"mAP50:      {metrics.box.map50:.4f}")
print(f"mAP50-95:   {metrics.box.map:.4f}")
print(f"Precision:  {metrics.box.mp:.4f}")
print(f"Recall:     {metrics.box.mr:.4f}")
print(f"\nConfusion matrix saved to: runs/detect/val*/confusion_matrix.png")
