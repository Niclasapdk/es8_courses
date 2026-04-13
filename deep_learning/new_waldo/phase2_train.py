import subprocess, sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"])

import os
import random
import shutil
import yaml
from ultralytics import YOLO

BASE     = "/kaggle/working"
IMG_SRC  = "/kaggle/input/datasets/johnask/waldo-images-with-negatives/resized_224x224"
DATASET  = os.path.join(BASE, "dataset")

LBL_SRC = "/kaggle/input/datasets/johnask/waldo-labels-v2/labels_resized_224x224"

# Sanity check
lbl_files = os.listdir(LBL_SRC)
print(f"Found {len(lbl_files)} label files — first: {lbl_files[0]}")
assert len(lbl_files) == 5500, f"Expected 5500 labels, got {len(lbl_files)}"

# ── Step 1: split 70/15/15 and copy into YOLO folder structure ──

random.seed(42)
names = sorted([f[:-4] for f in os.listdir(IMG_SRC) if f.endswith(".png")])
random.shuffle(names)

n = len(names)
splits = {
    "train": names[: int(0.70 * n)],
    "val":   names[int(0.70 * n) : int(0.85 * n)],
    "test":  names[int(0.85 * n) :],
}

for split, items in splits.items():
    img_dir = os.path.join(DATASET, "images", split)
    lbl_dir = os.path.join(DATASET, "labels", split)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)
    for name in items:
        shutil.copy2(os.path.join(IMG_SRC, name + ".png"), img_dir)
        shutil.copy2(os.path.join(LBL_SRC, name + ".txt"), lbl_dir)
    print(f"{split}: {len(items)} images")

# ── Step 2: write data.yaml ──

data_cfg = {
    "path": DATASET,
    "train": "images/train",
    "val":   "images/val",
    "test":  "images/test",
    "nc": 2,
    "names": ["waldo", "wilma"],
}
data_yaml = os.path.join(BASE, "data.yaml")
with open(data_yaml, "w") as f:
    yaml.dump(data_cfg, f, default_flow_style=False)
print(f"Wrote {data_yaml}")

# ── Step 3: train YOLOv8 nano ──

model = YOLO("yolov8s.pt")
model.train(
    data=data_yaml,
    epochs=30,
    imgsz=224,
    batch=32,
    workers=4,
    project=os.path.join(BASE, "runs"),
    name="waldo_wilma",
)
