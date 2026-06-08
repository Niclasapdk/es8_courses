# this script makes negative images no waldo no wilma and puts them
# together with the real images so YOLO learns what an empty scene looks like

import os
import shutil
import random
import numpy as np
from PIL import Image

random.seed(123)
np.random.seed(123)

# kaggle paths 
base = "/kaggle/input/datasets/sheshngupta/waldowilma/generated_images"
resized_dir = base + "/resized_224x224"

out = "/kaggle/working"
labels_dir = out + "/labels_resized_224x224" 

combined_imgs = out + "/images_combined"
combined_labels = out + "/labels_combined"

if not os.path.exists(combined_imgs):
    os.makedirs(combined_imgs)
if not os.path.exists(combined_labels):
    os.makedirs(combined_labels)

N_NEGATIVE = 500

# we take 50 random images and take the median pixel by pixel
print("Computing median background...")

all_img_files = []
for filename in os.listdir(resized_dir):
    if filename.endswith(".png"):
        all_img_files.append(resized_dir + "/" + filename)
all_img_files.sort()

sample = random.sample(all_img_files, 50)

img_list = []
for p in sample:
    pil_img = Image.open(p).convert("RGB")
    arr = np.array(pil_img, dtype=np.uint8)
    img_list.append(arr)

# stack them into one big array of shape (50, H, W, 3) and take the median
stack = np.stack(img_list, axis=0)
background = np.median(stack, axis=0).astype(np.uint8)

print("  Background shape:", background.shape)

print("")
print("Generating", N_NEGATIVE, "negative samples...")

for i in range(N_NEGATIVE):
    # work in float32 so we dont get overflow when adding noise
    img = background.astype(np.float32).copy()

    # add gaussian noise with random strength
    sigma = np.random.uniform(5, 15)
    noise = np.random.normal(0, sigma, img.shape)
    img = img + noise

    # add a random brightness offset
    brightness = np.random.uniform(-20, 20)
    img = img + brightness

    # adjust contrast
    mean = img.mean()
    contrast_factor = np.random.uniform(0.85, 1.15)
    img = (img - mean) * contrast_factor + mean

    # clip back into the valid 0-255 range and convert to uint8
    img = np.clip(img, 0, 255)
    img = img.astype(np.uint8)

    name = "negative_" + str(i).zfill(4)
    out_img_path = combined_imgs + "/" + name + ".png"
    Image.fromarray(img).save(out_img_path)

    out_label_path = combined_labels + "/" + name + ".txt"
    f = open(out_label_path, "w")
    f.close()

    if (i + 1) % 100 == 0:
        print("  done", i + 1, "/", N_NEGATIVE)


print("")
print("Copying true samples into combined folders")

true_imgs = []
for filename in os.listdir(resized_dir):
    if filename.endswith(".png"):
        true_imgs.append(filename)
true_imgs.sort()

missing_labels = 0

for img_name in true_imgs:
    src_img = resized_dir + "/" + img_name
    dst_img = combined_imgs + "/" + img_name

    stem = img_name[:-4]   # strip ".png"
    src_label = labels_dir + "/" + stem + ".txt"
    dst_label = combined_labels + "/" + stem + ".txt"

    shutil.copy(src_img, dst_img)

    if os.path.exists(src_label):
        shutil.copy(src_label, dst_label)
    else:
        f = open(dst_label, "w")
        f.close()
        missing_labels = missing_labels + 1

if missing_labels > 0:
    print("  WARNING:", missing_labels,
          "images had no matching label - written as negatives")


# count up everything
n_imgs = 0
for filename in os.listdir(combined_imgs):
    if filename.endswith(".png"):
        n_imgs = n_imgs + 1

n_labels = 0
n_empty = 0
for filename in os.listdir(combined_labels):
    if filename.endswith(".txt"):
        n_labels = n_labels + 1
        full_path = combined_labels + "/" + filename
        if os.path.getsize(full_path) == 0:
            n_empty = n_empty + 1

print("")
print("Combined dataset written to", out)
print("  Images :", n_imgs)
print("  Labels :", n_labels, "(", n_empty, "empty / negatives )")
print("  Mismatch:", abs(n_imgs - n_labels))
