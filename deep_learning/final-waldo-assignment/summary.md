# Summary of changes to `phase1LabelGeneration.py`

## The problem we hit

The first run of label generation produced a clearly skewed distribution
in `annotations_resized.csv`:

| set | waldo | wilma |
|------|-------|-------|
| original 350x500 | 2488 | 2512 |
| resized 224x224 (before fix) | 3160 | 1840 |

The original-resolution split is roughly 50/50 (matching the dataset),
but the resized split is way off. Since the two sets are the *same*
images at different resolutions, the labels should match. The
discrepancy meant `which_character` was misclassifying a chunk of the
224x224 images.

## Why it happens

`which_character` did an MSE comparison between the cropped character
and each template (`waldo.png`, `wilma.png`). Both characters wear
nearly identical red/white striped shirts — so most of the comparison
signal comes from a region that looks the same for both classes. The
real discriminator is the lower body:

- Waldo: dark blue denim jeans
- Wilma: bright cyan-blue skirt

At 224x224 the character bbox is only ~26x60 px. After the template
gets resized down to that and put through `cv2.resize` with
`INTER_AREA`, the cyan skirt color gets averaged into the surrounding
pixels and the signal is mostly gone. So the classifier defaults toward
Waldo.

## Fix #1 — bottom-half MSE

Replaced the full-crop MSE with an MSE that only looks at the bottom
half of the crop (and the bottom half of each template). The shirt
region was contributing noise, not signal — removing it makes the
comparison sharper.

Effect:
- original 350x500: 2488 / 2512  ← now balanced
- resized 224x224:  3058 / 1942  ← improved but still skewed

So the bottom-half trick fixed the high-res case but couldn't rescue
the low-res case. Resolution loss kills the skirt color too thoroughly
at 26x30 px.

## Fix #2 — share classifications across passes

The two image folders (`original_350x500/` and `resized_224x224/`)
contain the *same images* — every `image_N.png` has a matching
`resized_image_N.png`. So once we've correctly classified the
high-res version, we already know the answer for the low-res version.
There's no need to re-run `which_character` on the resized images at
all.

Changes:

- `process_folder` takes an optional `class_lookup` dict
  (`stem -> "waldo"|"wilma"`). When provided, it skips
  `which_character` and reuses the existing label.
- `process_folder` returns a `{stem: char_name}` dict at the end.
- `__main__` runs the original-resolution pass first, captures its
  output dict, prefixes the keys with `"resized_"` to match the
  resized filenames, and feeds it into the resized pass.

The bbox is still detected per-image (since the pixel coordinates
differ between resolutions and we want accurate boxes), but the
class identity is carried over from the reliable high-res pass.

## Functions touched

| function | change |
|----------|--------|
| `which_character` | full-crop MSE → bottom-half MSE only |
| `process_folder` | added `class_lookup` parameter, returns `{stem: char_name}` |
| `__main__` | captures `orig_classes`, builds `resized_lookup`, passes it to the resized call |

## How to verify after re-running

1. Run `phase1LabelGeneration.py` on Kaggle.
2. Check the counts again:
   ```
   grep -c waldo annotations_original.csv
   grep -c wilma annotations_original.csv
   grep -c waldo annotations_resized.csv
   grep -c wilma annotations_resized.csv
   ```
   Both files should show roughly 2500 / 2500.
3. Rebuild `labels_combined/`, retrain with `phase2Train.py`, and run
   `model_validation.py`. The output grid should have noticeably fewer
   red borders than before, and the predicted classes should align
   with the actual character in each image.

## Caveats

- The `f"resized_{stem}"` mapping assumes the originals are named
  `image_N.png` and the resized ones are `resized_image_N.png`. If the
  Kaggle dataset uses a different convention, the lookup will silently
  miss and fall back to running `which_character` on each image. Quick
  check before re-running:
  `ls /kaggle/input/.../original_350x500/ | head`
- Bbox detection is still done independently for each resolution, so
  any errors from `get_bbox` (e.g. when the background subtraction
  fails) are unaffected by these changes.
