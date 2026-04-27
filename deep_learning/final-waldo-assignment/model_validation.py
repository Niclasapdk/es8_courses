"""
Local verification of the trained YOLOv8 model.

1. Download best.pt from Kaggle and place it in new_waldo/
2. Run: python verify_model.py
3. Open verify_model_output.png to see results
   - Green border = prediction matches ground truth
   - Red border   = wrong prediction
"""
import os
import random
import cv2
import numpy as np
from ultralytics import YOLO

BASE      = os.path.dirname(os.path.abspath(__file__))
WEIGHTS   = os.path.join(BASE, "best.pt")
IMG_DIR   = os.path.join(BASE, "images_combined")
LBL_DIR   = os.path.join(BASE, "labels_combined")
OUT_FILE  = os.path.join(BASE, "verify_model_output.png")
N_SAMPLES = 16
CLASSES   = ["waldo", "wilma"]
COLORS    = {"waldo": (0, 0, 255), "wilma": (255, 0, 0)}  # BGR

if not os.path.exists(WEIGHTS):
    raise FileNotFoundError(f"Put best.pt in {BASE}  (download from Kaggle runs/waldo_wilma/weights/)")

model = YOLO(WEIGHTS)

all_imgs = sorted([f for f in os.listdir(IMG_DIR) if f.endswith(".png")])
random.seed(0)
sample = random.sample(all_imgs, min(N_SAMPLES, len(all_imgs)))

cols   = 4
rows   = (len(sample) + cols - 1) // cols
cell   = 224
canvas = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)

correct = 0

for idx, fname in enumerate(sample):
    img_path = os.path.join(IMG_DIR, fname)
    lbl_path = os.path.join(LBL_DIR, fname.replace(".png", ".txt"))

    img = cv2.imread(img_path)

    # ground truth class
    gt_class = None
    if os.path.exists(lbl_path):
        parts = open(lbl_path).read().strip().split()
        if parts:
            gt_class = int(parts[0])

    # inference — low conf threshold to catch anything the model detects
    results    = model(img_path, verbose=False, conf=0.1)[0]
    pred_class = None
    pred_conf  = 0.0
    pred_box   = None
    if len(results.boxes):
        best       = results.boxes[results.boxes.conf.argmax()]
        pred_class = int(best.cls.item())
        pred_conf  = float(best.conf.item())
        x1, y1, x2, y2 = map(int, best.xyxy[0].tolist())
        pred_box   = (x1, y1, x2, y2)

    vis = img.copy()
    if pred_box:
        color = COLORS.get(CLASSES[pred_class], (200, 200, 200))
        cv2.rectangle(vis, pred_box[:2], pred_box[2:], color, 2)
        cv2.putText(vis, f"{CLASSES[pred_class]} {pred_conf:.2f}",
                    (pred_box[0], max(pred_box[1] - 5, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    match = pred_class == gt_class if (pred_class is not None and gt_class is not None) else False
    if match:
        correct += 1

    border = (0, 200, 0) if match else (0, 0, 200)
    vis = cv2.copyMakeBorder(vis, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=border)
    vis = cv2.resize(vis, (cell, cell))

    r, c = divmod(idx, cols)
    canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = vis

cv2.imwrite(OUT_FILE, canvas)
print(f"Saved {OUT_FILE}")
print(f"Spot-check accuracy: {correct}/{len(sample)} = {correct/len(sample)*100:.1f}%")
print("\nIf accuracy is 0% the model likely trained without valid labels — retrain with the fixed phase2_train.py")
