"""
Isolating cortex from a myelin section, evaluated with the Dice score
====================================================================

These myelin sections have a BRIGHT background, PALE cortex, and DARK white
matter. This script isolates the cortex and then measures how good the result
is against a manual segmentation, using the Dice score.

Pipeline:
  1. Flip the image intensities (bright background -> dark).
  2. Isolate the foreground (tissue) by thresholding away the background,
     using the image's MODAL value (most common pixel) as the background.
  3. Keep only the two largest connected regions (the two hemispheres).
  4. OTSU: threshold inside the tissue to isolate cortex (kept BELOW threshold,
     because flipping made the pale cortex the darker class).
  5. DICE: compare the Otsu cortex mask against a manual segmentation and report
     a Dice score.

You provide TWO files: the raw section image, and your manual segmentation of
the cortex (a black/white image, white = cortex).

Run:  python cortex_pipeline.py
"""

import imageio.v3 as iio
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

IMAGE_PATH = "redroo.jpg"                 # raw section
MANUAL_PATH = "redroo_manual.png"         # your manual cortex segmentation
OUTPUT_PATH = "cortex_pipeline.png"


# ---------------------------------------------------------------------------
# Load the raw image (grayscale).
# ---------------------------------------------------------------------------
img = iio.imread(IMAGE_PATH, mode="L").astype(np.uint8)
print("Loaded image:", img.shape)


# ===========================================================================
# STEP 1 - Flip the image intensities
# ===========================================================================
# Background is bright (~255); later steps are easier when background is dark.
# Inverting: every value v becomes (255 - v).
flipped = 255 - img


# ===========================================================================
# STEP 2 - Isolate the foreground by thresholding away the background
# ===========================================================================
# The background dominates the frame, so the MODAL pixel value (most common) is
# a good estimate of the background brightness.
values, counts = np.unique(img, return_counts=True)
background_value = values[np.argmax(counts)]
print("Modal (background) value:", background_value)

# In the flipped image the background sits near (255 - background_value); tissue
# is clearly brighter. "+ 15" is a small margin above background noise.
foreground = flipped > (255 - background_value + 15)
print("Foreground fraction:", round(foreground.mean(), 3))


# ===========================================================================
# STEP 3 - Keep only the two largest connected components (the hemispheres)
# ===========================================================================
labels, num_features = ndi.label(foreground)
print("Connected components found:", num_features)
sizes = ndi.sum(np.ones_like(labels), labels, range(1, num_features + 1))
two_largest_ids = np.argsort(sizes)[::-1][:2] + 1     # +1: labels start at 1
tissue = np.isin(labels, two_largest_ids)
print("Two-largest fraction:", round(tissue.mean(), 3))


# ===========================================================================
# STEP 4 - Otsu threshold inside the tissue to isolate cortex
# ===========================================================================
otsu_threshold = threshold_otsu(flipped[tissue])
print("Otsu threshold within tissue:", otsu_threshold)
# Flipped -> pale cortex is now the DARKER class -> keep pixels BELOW threshold.
cortex_otsu = tissue & (flipped < otsu_threshold)
print("Otsu cortex fraction:", round(cortex_otsu.mean(), 3))


# ===========================================================================
# STEP 5 - Evaluate against the manual segmentation (Dice score)
# ===========================================================================
# Load the manual segmentation and turn it into a True/False mask. A hand-drawn
# mask is usually white (255) on black (0); "> 0" makes any non-black pixel
# True. Loading as grayscale guards against colour/alpha-channel surprises.
manual = iio.imread(MANUAL_PATH, mode="L")
manual_mask = manual > 0
print("\nManual segmentation fraction:", round(manual_mask.mean(), 3))

def dice_score(pred, truth):
    """Dice score between two boolean masks.

    Dice = 2 * (overlap) / (size of pred + size of truth)

    It ranges from 0 (no overlap) to 1 (perfect overlap). A standard way to
    measure how well a predicted mask matches a reference mask.
    """
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    overlap = np.logical_and(pred, truth).sum()      # pixels in BOTH
    return 2.0 * overlap / (pred.sum() + truth.sum())

dice_otsu = dice_score(cortex_otsu, manual_mask)
print("Dice (Otsu vs manual):", round(dice_otsu, 3))


# ===========================================================================
# Output figure: original + each step + Otsu mask vs manual
# ===========================================================================
fig, ax = plt.subplots(2, 3, figsize=(14, 9))
ax = ax.ravel()

ax[0].imshow(img, cmap="gray");             ax[0].set_title("Original")
ax[1].imshow(flipped, cmap="gray");         ax[1].set_title("1. Flipped intensities")
ax[2].imshow(foreground, cmap="gray");      ax[2].set_title("2. Foreground (bg threshold)")
ax[3].imshow(tissue, cmap="gray");          ax[3].set_title("3. Two largest components")
ax[4].imshow(cortex_otsu, cmap="gray");     ax[4].set_title(f"4. Cortex - Otsu\nDice={dice_otsu:.3f}")

# last panel: overlay Otsu (red) vs manual (green); overlap shows as yellow
overlay = np.zeros((*img.shape, 3))
overlay[..., 0] = cortex_otsu       # red   = Otsu prediction
overlay[..., 1] = manual_mask       # green = manual truth
ax[5].imshow(overlay);              ax[5].set_title("Otsu (red) vs manual (green)\noverlap = yellow")

for a in ax:
    a.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150)
plt.close()
print("\nSaved figure to:", OUTPUT_PATH)
