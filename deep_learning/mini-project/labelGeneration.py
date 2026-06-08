# label generation for waldo/wilma dataset
# we subtract the background from each image to find where the character is
# then figure out if its waldo or wilma by comparing to the template images
# outputs YOLO format labels

import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import csv
import random
import os

base = Path("/kaggle/input/datasets/sheshngupta/waldowilma/generated_images")
char_dir = base / "charecters"
orig_dir = base / "original_350x500"
resized_dir = base / "resized_224x224"

path_out = Path("/kaggle/working")
labels_orig = path_out / "labels_original_350x500"
labels_resized = path_out / "labels_resized_224x224"
os.makedirs(labels_orig, exist_ok=True)
os.makedirs(labels_resized, exist_ok=True)

CLASS_MAP = {"waldo": 0, "wilma": 1}

DIFF_THRESH = 30  # how different a pixel needs to be to count


def get_background(img_dir, num=50):
    # take images and compute the median to get the background
    all_imgs = sorted(img_dir.glob("*.png"))
    sample = random.sample(all_imgs, num)

    imgs = []
    for p in sample:
        im = np.array(Image.open(str(p)).convert("RGB"), dtype=np.uint8)
        imgs.append(im)

    stacked = np.stack(imgs, axis=0)
    bg = np.median(stacked, axis=0).astype(np.uint8)
    return bg


def get_bbox(img, bg):
    # subtract background from image threshold it find the biggest blob
    diff = np.abs(img.astype(np.int16) - bg.astype(np.int16))

    # take max across RGB channels
    diff_gray = diff.max(axis=2).astype(np.uint8)

    _, mask = cv2.threshold(diff_gray, DIFF_THRESH, 255, cv2.THRESH_BINARY)

    # morphological to clean up noise
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    biggest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(biggest)


def which_character(crop, templates):
    # compare the crop to each template using RGB MSE on the bottom half
    h, w = crop.shape[:2]
    split = h // 2
    best = None
    best_err = 999999999

    for name, (rgb, alpha) in templates.items():
        # resize template to match crop size
        tmp_resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
        alpha_resized = cv2.resize(alpha, (w, h), interpolation=cv2.INTER_AREA)

        # slice bottom half of crop, template, and alpha mask
        crop_b = crop[split:, :]
        tmp_b = tmp_resized[split:, :]
        alpha_b = alpha_resized[split:, :]

        # only look at non-transparent pixels
        valid = alpha_b > 128
        if valid.sum() == 0:
            continue

        d = crop_b.astype(np.float32) - tmp_b.astype(np.float32)
        mse = (d[valid] ** 2).mean()

        if mse < best_err:
            best_err = mse
            best = name

    return best


def load_template(path):
    img = Image.open(str(path)).convert("RGBA")
    arr = np.array(img)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    return rgb, alpha


def process_folder(img_dir, label_dir, csv_path, bg, templates, class_lookup=None):
    # go through all images in a folder and generate labels
    img_paths = sorted(img_dir.glob("*.png"))
    print(f"Processing {len(img_paths)} images from {img_dir.name}...")

    results = []
    no_detect = 0
    classes_out = {}  # stem -> char_name, returned at the end

    for idx, img_path in enumerate(img_paths):
        img = np.array(Image.open(str(img_path)).convert("RGB"), dtype=np.uint8)
        img_h, img_w = img.shape[:2]

        bbox = get_bbox(img, bg)

        if bbox is None:
            x, y, w, h = img_w // 4, img_h // 4, img_w // 2, img_h // 2
            char_name = "unknown"
            cls = -1
            no_detect += 1
        else:
            x, y, w, h = bbox
            # make sure we dont go out of bounds
            x = max(0, x)
            y = max(0, y)
            w = min(w, img_w - x)
            h = min(h, img_h - y)

            if class_lookup is not None and img_path.stem in class_lookup:
                # reuse the class we already figured out from the high-res pass
                char_name = class_lookup[img_path.stem]
            else:
                crop = img[y:y+h, x:x+w]
                char_name = which_character(crop, templates)
            cls = CLASS_MAP[char_name]

        classes_out[img_path.stem] = char_name

        # convert to YOLO format with normalized center coords and width/height
        cx = (x + w / 2) / img_w
        cy = (y + h / 2) / img_h
        nw = w / img_w
        nh = h / img_h

        # write label file
        label_path = label_dir / (img_path.stem + ".txt")
        with open(label_path, "w") as f:
            f.write(f"{cls} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")

        results.append({
            "filename": img_path.name,
            "character": char_name,
            "class_id": cls,
            "x": x, "y": y, "w": w, "h": h,
            "img_w": img_w, "img_h": img_h,
            "cx": f"{cx:.6f}", "cy": f"{cy:.6f}",
            "nw": f"{nw:.6f}", "nh": f"{nh:.6f}",
        })

        # print progress every 500 images
        if (idx + 1) % 500 == 0:
            print(f"  done {idx + 1} / {len(img_paths)}")

    # save csv
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"  done! {no_detect} images had no detection")
    return classes_out


if __name__ == "__main__":
    print("loading templates...")
    templates = {}
    for name in CLASS_MAP:
        templates[name] = load_template(char_dir / f"{name}.png")
        rgb, _ = templates[name]
        print(f"  {name}: {rgb.shape[1]}x{rgb.shape[0]}")

    # original 350x500
    print("\ncomputing background for original set...")
    bg_orig = get_background(orig_dir, num=50)
    orig_classes = process_folder(
        orig_dir, labels_orig, path_out / "annotations_original.csv",
        bg_orig, templates,
    )

    # resized 224x224
    resized_lookup = {f"resized_{stem}": name for stem, name in orig_classes.items()}
    resized_lookup.update(orig_classes)

    print("\ncomputing background for resized set...")
    bg_resized = get_background(resized_dir, num=50)
    process_folder(
        resized_dir, labels_resized, path_out / "annotations_resized.csv",
        bg_resized, templates,
        class_lookup=resized_lookup,
    )

    print("\nall done!")
    print(f"labels saved to {labels_orig} and {labels_resized}")

