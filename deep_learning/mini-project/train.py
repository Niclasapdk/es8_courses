# trains a YOLOv8 model on the combined waldo/wilma dataset.
# run this after generateNegativeSamples.py so we have images_combined/
# and labels_combined/

import os
import random
import shutil
import yaml
import subprocess, sys 
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "ultralytics"]) # needed for kaggle
from ultralytics import YOLO

base = "/kaggle/working"
dataset = os.path.join(base, "dataset")
img_src = "/kaggle/working/images_combined"
lbl_src = "/kaggle/working/labels_combined"

random.seed(42)

# get all the image filenames
names = []
for filename in os.listdir(img_src):
    if filename.endswith(".png"):
        names.append(filename[:-4])
names.sort()
random.shuffle(names)

n = len(names)
train_end = int(0.70 * n)
val_end = int(0.85 * n)

train_names = names[:train_end]
val_names = names[train_end:val_end]
test_names = names[val_end:]

print("train:", len(train_names), "images")
print("val:  ", len(val_names), "images")
print("test: ", len(test_names), "images")


# YOLO expects this folder layout:
#   dataset/images/train, dataset/images/val, dataset/images/test
#   dataset/labels/train, dataset/labels/val, dataset/labels/test
splits = {
    "train": train_names,
    "val": val_names,
    "test": test_names,
}

for split_name in splits:
    items = splits[split_name]

    img_dir = os.path.join(dataset, "images", split_name)
    lbl_dir = os.path.join(dataset, "labels", split_name)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lbl_dir, exist_ok=True)

    # copy each image and its label file into the right split folder
    for name in items:
        src_img = os.path.join(img_src, name + ".png")
        src_lbl = os.path.join(lbl_src, name + ".txt")
        shutil.copy(src_img, img_dir)
        shutil.copy(src_lbl, lbl_dir)


# count how many "negatives" ended up in each split
# so we can confirm the negatives got spread across train/val/test
for split_name in ["train", "val", "test"]:
    lbl_dir = os.path.join(dataset, "labels", split_name)
    files = os.listdir(lbl_dir)

    empty = 0
    for f in files:
        full_path = os.path.join(lbl_dir, f)
        if os.path.getsize(full_path) == 0:
            empty = empty + 1

    pct = 100 * empty / len(files)
    print(split_name, ":", len(files), "labels,", empty, "negatives (",
          round(pct, 1), "%)")

data_cfg = {
    "path": dataset,
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "nc": 2,
    "names": ["waldo", "wilma"],   # class 0 = waldo, class 1 = wilma
}

data_yaml = os.path.join(base, "data.yaml")
with open(data_yaml, "w") as f:
    yaml.dump(data_cfg, f, default_flow_style=False)
print("Wrote", data_yaml)

# yolov8s = the small variant
model = YOLO("yolov8s.pt")
model.train(
    data=data_yaml,
    epochs=30,
    imgsz=224,
    batch=32,
    workers=4,
    project=os.path.join(base, "runs"),
    name="waldo_wilma",
)
