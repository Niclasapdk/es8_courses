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
from tqdm import tqdm
import os

# paths (change these if ur running locally)
BASE = Path("/kaggle/input/datasets/sheshngupta/waldowilma/generated_images")
CHAR_DIR = BASE / "charecters"
ORIG_DIR = BASE / "original_350x500"
RESIZED_DIR = BASE / "resized_224x224"

OUT = Path("/kaggle/working")
LABELS_ORIG = OUT / "labels_original_350x500"
LABELS_RESIZED = OUT / "labels_resized_224x224"
os.makedirs(LABELS_ORIG, exist_ok=True)
os.makedirs(LABELS_RESIZED, exist_ok=True)

# waldo = 0, wilma = 1
CLASS_MAP = {"waldo": 0, "wilma": 1}

DIFF_THRESH = 30  # how different a pixel needs to be to count


def get_background(img_dir, num=50):
    """
    take images and compute the median to get the background
    this works because each image has the character in a different spot
    so the median will just be the background with no character
    """
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
    """
    subtract background from image, threshold it, find the biggest blob
    returns (x, y, w, h) or None if nothing found
    """
    # need int16 so we dont get overflow issues with subtraction
    diff = np.abs(img.astype(np.int16) - bg.astype(np.int16))

    # take max across RGB channels
    diff_gray = diff.max(axis=2).astype(np.uint8)

    _, mask = cv2.threshold(diff_gray, DIFF_THRESH, 255, cv2.THRESH_BINARY)

    # morphological stuff to clean up noise
    kern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kern, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kern, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return None

    # just grab the biggest one, should be the character
    biggest = max(contours, key=cv2.contourArea)
    return cv2.boundingRect(biggest)


def which_character(crop, templates):
    """
    compare the crop to each template using RGB MSE on the BOTTOM HALF only.

    why bottom half: waldo and wilma have nearly identical red/white striped
    shirts in the upper body. the discriminative region is the lower body
    (wilma's cyan skirt vs waldo's dark jeans), so comparing the full crop
    drowns the signal in shirt noise.

    NOTE: we only compare pixels where the template isnt transparent
    otherwise the background pixels mess up the comparison
    """
    h, w = crop.shape[:2]
    split = h // 2  # midpoint — everything below this is "bottom half"
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
    """load character template, return RGB + alpha channel"""
    img = Image.open(str(path)).convert("RGBA")
    arr = np.array(img)
    rgb = arr[:, :, :3]
    alpha = arr[:, :, 3]
    return rgb, alpha


def process_folder(img_dir, label_dir, csv_path, bg, templates, class_lookup=None):
    """
    go through all images in a folder and generate labels.

    if class_lookup is given (dict of stem -> char_name) we skip the
    template-matching step and just reuse those classes. needed because
    which_character is unreliable on the 224x224 images (skirt color
    gets washed out at low res), but works fine on the 350x500 ones.
    so we classify on the originals and reuse the answer for the resized.
    """
    img_paths = sorted(img_dir.glob("*.png"))
    print(f"Processing {len(img_paths)} images from {img_dir.name}...")

    results = []
    no_detect = 0
    classes_out = {}  # stem -> char_name, returned at the end

    for img_path in tqdm(img_paths):
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

        # convert to YOLO format (normalized center coords + width/height)
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
        templates[name] = load_template(CHAR_DIR / f"{name}.png")
        rgb, _ = templates[name]
        print(f"  {name}: {rgb.shape[1]}x{rgb.shape[0]}")

    # --- original 350x500 ---
    # do this one first so we get reliable classifications to feed
    # into the resized pass
    print("\ncomputing background for original set...")
    bg_orig = get_background(ORIG_DIR, num=50)
    orig_classes = process_folder(
        ORIG_DIR, LABELS_ORIG, OUT / "annotations_original.csv",
        bg_orig, templates,
    )

    # --- resized 224x224 ---
    # the resized files are named "resized_image_N" while the originals are
    # "image_N". build a lookup keyed on the resized stem so the second pass
    # can find it. also include the original keys as a fallback in case the
    # naming convention is different than expected.
    resized_lookup = {f"resized_{stem}": name for stem, name in orig_classes.items()}
    resized_lookup.update(orig_classes)

    print("\ncomputing background for resized set...")
    bg_resized = get_background(RESIZED_DIR, num=50)
    process_folder(
        RESIZED_DIR, LABELS_RESIZED, OUT / "annotations_resized.csv",
        bg_resized, templates,
        class_lookup=resized_lookup,
    )

    print("\nall done!")
    print(f"labels saved to {LABELS_ORIG} and {LABELS_RESIZED}")

