"""
Segmenting cortex from a two-layer annotation TIFF, evaluated with Dice
======================================================================

Your annotation is a single TIFF with TWO layers:
  - TOP layer  (page 0): your manual segmentation (two values: fg / bg)
  - BOTTOM layer (page 1): the raw section image

This script reads both layers from that one file, isolates the cortex from the
raw image automatically, and measures how well the automatic result matches
your manual segmentation using the Dice score.

Pipeline:
  0. Read the two layers from the annotation TIFF.
  1. Flip the raw image intensities (bright background -> dark) so Nissl and
     myelin sections can be handled the same way.
  2. Isolate the foreground (tissue) by thresholding away the background,
     using the image's MODAL value (most common pixel) as the background.
  3. Keep only the two largest connected regions (the two hemispheres).
  4. OTSU: threshold inside the tissue to isolate cortex (kept BELOW threshold,
     because flipping made the pale cortex the darker class).
  5. DICE: compare the automatic cortex mask against the manual layer.

Run:  python cortex_pipeline.py path/to/annotation.tif
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

OUTPUT_PATH = "cortex_pipeline.png"


# ===========================================================================
# STEP 0 - Read the two layers from the annotation TIFF
# ===========================================================================
def read_annotation(tif_path):
    """Open a two-layer annotation TIFF and return its layers as arrays.

    Returns
    -------
    seg_layer : np.ndarray   the manual segmentation (top layer, page 0)
    raw_layer : np.ndarray   the raw section image  (bottom layer, page 1)
    """
    with tifffile.TiffFile(tif_path) as tf:
        seg_layer = tf.pages[0].asarray()
        raw_layer = tf.pages[1].asarray()
    return seg_layer, raw_layer


# take the annotation path from the command line (default for quick testing)
tif_path = sys.argv[1] if len(sys.argv) > 1 else "annotation_example.tif"
manual_layer, img = read_annotation(tif_path)
img = img.astype(np.uint8)
manual_mask = manual_layer > 0        # foreground = any non-zero pixel
print("Read annotation:", tif_path)
print("  raw layer:", img.shape, " manual layer:", manual_layer.shape)


# ===========================================================================
# STEP 1 - Flip the image intensities
# ===========================================================================
# Nissl GM is dark, myelin GM is bright. Flipping makes both behave the same.
# img.max() - img inverts without assuming an 8-bit range.
flipped = img.max() - img


# ===========================================================================
# STEP 2 - Isolate the foreground by thresholding away the background
# ===========================================================================
values, counts = np.unique(img, return_counts=True)
background_value = values[np.argmax(counts)]
print("Modal (background) value:", background_value)
foreground = flipped > (img.max() - background_value + 15)
print("Foreground fraction:", round(foreground.mean(), 3))


# ===========================================================================
# STEP 3 - Keep only the two largest connected components (the hemispheres)
# ===========================================================================
labels, num_features = ndi.label(foreground)
print("Connected components found:", num_features)
sizes = ndi.sum(np.ones_like(labels), labels, range(1, num_features + 1))
two_largest_ids = np.argsort(sizes)[::-1][:2] + 1
tissue = np.isin(labels, two_largest_ids)
print("Two-largest fraction:", round(tissue.mean(), 3))


# ===========================================================================
# STEP 4 - Otsu threshold inside the tissue to isolate cortex
# ===========================================================================
otsu_threshold = threshold_otsu(flipped[tissue])
print("Otsu threshold within tissue:", otsu_threshold)
cortex = tissue & (flipped < otsu_threshold)   # cortex = darker class (flipped)
print("Cortex fraction:", round(cortex.mean(), 3))


# ===========================================================================
# STEP 5 - Evaluate against the manual layer (Dice score)
# ===========================================================================
def dice_score(pred, truth):
    """Dice = 2 * overlap / (size of pred + size of truth). 0 = none, 1 = perfect."""
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    overlap = np.logical_and(pred, truth).sum()
    return 2.0 * overlap / (pred.sum() + truth.sum())

dice = dice_score(cortex, manual_mask)
print("\nDice (automatic vs manual):", round(dice, 3))


# ===========================================================================
# Output figure
# ===========================================================================
fig, ax = plt.subplots(2, 3, figsize=(14, 9))
ax = ax.ravel()
ax[0].imshow(img, cmap="gray");         ax[0].set_title("Raw (from TIFF)")
ax[1].imshow(flipped, cmap="gray");     ax[1].set_title("1. Flipped intensities")
ax[2].imshow(foreground, cmap="gray");  ax[2].set_title("2. Foreground (bg threshold)")
ax[3].imshow(tissue, cmap="gray");      ax[3].set_title("3. Two largest components")
ax[4].imshow(cortex, cmap="gray");      ax[4].set_title(f"4. Cortex - Otsu\nDice={dice:.3f}")
overlay = np.zeros((*img.shape, 3))
overlay[..., 0] = cortex               # red   = automatic
overlay[..., 1] = manual_mask          # green = manual
ax[5].imshow(overlay);                 ax[5].set_title("auto (red) vs manual (green)\noverlap = yellow")
for a in ax:
    a.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
plt.close()
print("Saved figure to:", OUTPUT_PATH)
