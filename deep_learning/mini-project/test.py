# test script for the trained YOLOv8 waldo and wilma model
# green border means prediction was correct or model correctly found nothing
# red border means wrong prediction or no label file to compare against

import os
import cv2
import numpy as np
from ultralytics import YOLO

base = os.path.dirname(os.path.abspath(__file__))
weights = os.path.join(base, "best.pt")
img_dir = os.path.join(base, "demo_images")
lbl_dir = os.path.join(base, "demo_labels")
out_file = os.path.join(base, "demo_output.png")

classes = ["waldo", "wilma"]
colors = {"waldo": (0, 0, 255), "wilma": (255, 0, 0)}

print("Loading model from", weights)
model = YOLO(weights)

demo_imgs = []
for filename in os.listdir(img_dir):
    if filename.endswith(".png") or filename.endswith(".jpg"):
        demo_imgs.append(filename)
demo_imgs.sort()

# arrange the images in a grid 4 per row
cols = 4
rows = 3

cell = 224   # each grid cell is 224x224 pixels
canvas = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)

correct = 0
have_labels = 0

for idx in range(len(demo_imgs)):
    fname = demo_imgs[idx]
    img_path = os.path.join(img_dir, fname)
    stem = fname[:-4]
    lbl_path = os.path.join(lbl_dir, stem + ".txt")

    img = cv2.imread(img_path)

    gt_class = None
    gt_is_negative = False
    if os.path.exists(lbl_path):
        f = open(lbl_path)
        parts = f.read().strip().split()
        f.close()
        if len(parts) > 0:
            gt_class = int(parts[0])
        else:
            gt_is_negative = True
        have_labels = have_labels + 1

    # run YOLO on the image
    results = model(img_path, verbose=False, conf=0.1)[0]

    # find the box with the highest confidence
    pred_class = None
    pred_conf = 0.0
    pred_box = None
    if len(results.boxes) > 0:
        best_idx = 0
        best_conf = float(results.boxes[0].conf.item())
        for i in range(1, len(results.boxes)):
            c = float(results.boxes[i].conf.item())
            if c > best_conf:
                best_conf = c
                best_idx = i

        best = results.boxes[best_idx]
        pred_class = int(best.cls.item())
        pred_conf = float(best.conf.item())

        # xyxy is the bounding box as [x1, y1, x2, y2]
        coords = best.xyxy[0].tolist()
        x1 = int(coords[0])
        y1 = int(coords[1])
        x2 = int(coords[2])
        y2 = int(coords[3])
        pred_box = (x1, y1, x2, y2)

    # draw the prediction box on top of the image
    vis = img.copy()
    if pred_box is not None:
        x1, y1, x2, y2 = pred_box
        color = colors[classes[pred_class]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label_text = classes[pred_class] + " " + str(round(pred_conf, 2))
        cv2.putText(
            vis, label_text,
            (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1,
        )

    # decide border color
    is_correct = False
    if pred_class is not None and gt_class is not None and pred_class == gt_class:
        is_correct = True
    if pred_class is None and gt_is_negative:
        is_correct = True

    if is_correct:
        border = (0, 200, 0)
        correct = correct + 1
    else:
        border = (0, 0, 200)

    vis = cv2.copyMakeBorder(vis, 3, 3, 3, 3, cv2.BORDER_CONSTANT, value=border)
    vis = cv2.resize(vis, (cell, cell))

    r = idx // cols
    c = idx % cols
    canvas[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell] = vis


cv2.imwrite(out_file, canvas)
print("")
print("Saved visualization to", out_file)
