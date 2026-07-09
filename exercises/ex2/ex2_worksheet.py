"""
Cortex segmentation from a two-layer annotation TIFF -- WORKSHEET
================================================================

Your annotation is a single TIFF file with TWO layers:
  - TOP layer  (page 0): your manual segmentation (two values: foreground /
    background)
  - BOTTOM layer (page 1): the raw section image

Your job: read both layers from that one file, automatically isolate the cortex
from the raw image, and measure how well it matches your manual segmentation
using the Dice score.

Run it from the command line, passing your annotation file:

    python cortex_pipeline_worksheet.py path/to/annotation.tif

If you get stuck, a full solution is in `cortex_pipeline.py` -- but try the
hints first. When something breaks, READ THE ERROR (last line = what, line
number = where).
"""

# ---------------------------------------------------------------------------
# SETUP - imports.
#   - sys                                     (to read the file path argument)
#   - numpy (as np), matplotlib.pyplot (as plt)
#   - tifffile                                (to open the TIFF)
#   - from scipy import ndimage as ndi        (connected components)
#   - from skimage.filters import threshold_otsu
# ---------------------------------------------------------------------------

# --- your imports here ---

# OUTPUT_PATH = "cortex_pipeline.png"


# ===========================================================================
# STEP 0 - Write a function to read the two layers from the annotation TIFF
# ===========================================================================
# TASK: complete read_annotation() so it opens a two-layer TIFF and returns the
# segmentation layer and the raw layer as two separate NumPy arrays.
#
# Why a function? You'll reuse this every time you load an annotation, and later
# you may loop over many files -- a function keeps that clean.
#
# Hints:
#   - A multi-layer TIFF is a stack of "pages". tifffile reads them like this:
#         with tifffile.TiffFile(tif_path) as tf:
#             first_layer  = tf.pages[0].asarray()
#             second_layer = tf.pages[1].asarray()
#   - By our convention the TOP layer (page 0) is the SEGMENTATION and the
#     BOTTOM layer (page 1) is the RAW image. Return them in that order.
#   - Think about what each `.asarray()` gives you: a 2-D NumPy array of pixel
#     values, exactly like the images you loaded in Exercise 1.

def read_annotation(tif_path):
    """Open a two-layer annotation TIFF and return (seg_layer, raw_layer)."""
    # TODO: open the TIFF and pull out page 0 (segmentation) and page 1 (raw).
    seg_layer = None   # <-- replace this
    raw_layer = None   # <-- replace this
    return seg_layer, raw_layer


# Read the annotation path from the command line, then load the two layers.
# Hint:
#   tif_path = sys.argv[1]                       # the path you typed after the script name
#   manual_layer, img = read_annotation(tif_path)
#   img = img.astype("uint8")
#   manual_mask = manual_layer > 0               # foreground = any non-zero pixel

# --- your code here ---


# ===========================================================================
# STEP 1 - Flip the image intensities
# ===========================================================================
# Nissl grey matter is DARK; myelin grey matter is BRIGHT. Flipping the raw
# image makes both section types behave the same way for the steps below.
#
# Hint: subtract the image from its own maximum (works for any bit depth):
#     flipped = img.max() - img

# --- your code here ---


# ===========================================================================
# STEP 2 - Isolate the foreground by thresholding away the background
# ===========================================================================
# TASK: make a True/False mask `foreground` that is True on tissue.
#
# Hints:
#   - The background fills most of the frame, so the MOST COMMON pixel value
#     estimates the background brightness:
#         values, counts = np.unique(img, return_counts=True)
#         background_value = values[np.argmax(counts)]
#   - In the FLIPPED image, tissue is brighter than the flipped background:
#         foreground = flipped > (img.max() - background_value + 15)
#     The "+ 15" is a small margin above background noise.

# --- your code here ---


# ===========================================================================
# STEP 3 - Keep only the two largest connected components (the hemispheres)
# ===========================================================================
# TASK: remove text/scale-bar/specks by keeping only the two biggest blobs, in
# a mask called `tissue`.
#
# Hints:
#   - labels, num_features = ndi.label(foreground)
#   - sizes = ndi.sum(np.ones_like(labels), labels, range(1, num_features + 1))
#   - two_largest_ids = np.argsort(sizes)[::-1][:2] + 1   # +1: ids start at 1
#   - tissue = np.isin(labels, two_largest_ids)

# --- your code here ---


# ===========================================================================
# STEP 4 - Otsu threshold inside the tissue to isolate cortex
# ===========================================================================
# TASK: split tissue into cortex vs white matter with Otsu; keep cortex.
#
# Hints:
#   - otsu_threshold = threshold_otsu(flipped[tissue])
#   - Because you flipped, cortex is the DARKER class -> keep BELOW threshold:
#         cortex = tissue & (flipped < otsu_threshold)

# --- your code here ---


# ===========================================================================
# STEP 5 - Evaluate with the Dice score
# ===========================================================================
# Dice measures how well the automatic mask matches your manual one:
#     Dice = 2 * (pixels in BOTH) / (pixels in pred + pixels in truth)
# 0 = no overlap, 1 = perfect match.
#
# TASK: finish dice_score(), then print the Dice of `cortex` vs `manual_mask`.
#
# Hints:
#   - "pixels in both" = np.logical_and(pred, truth).sum()
#   - "pixels in a mask" = that mask's .sum()  (True counts as 1)

def dice_score(pred, truth):
    """Return the Dice score between two boolean masks (0 = none, 1 = perfect)."""
    pred = pred.astype(bool)
    truth = truth.astype(bool)

    # TODO: count the pixels True in BOTH masks (hint: np.logical_and(...).sum())
    overlap = 0     # <-- replace this

    # TODO: return 2 * overlap / (pred.sum() + truth.sum())
    return 0.0      # <-- replace this

# --- print the Dice score here ---
# dice = dice_score(cortex, manual_mask)
# print("Dice (automatic vs manual):", dice)


# ===========================================================================
# STEP 6 - Make an output figure
# ===========================================================================
# TASK: save a figure of the raw image, each step, and the Dice score in a title.
#
# Hints:
#   - fig, ax = plt.subplots(2, 3, figsize=(14, 9)); ax = ax.ravel()
#   - ax[0].imshow(img, cmap="gray"); ax[0].set_title("Raw")
#   - Put Dice in a title with an f-string: f"Cortex  Dice={dice:.3f}"
#   - for a in ax: a.axis("off");  then plt.savefig(OUTPUT_PATH, dpi=150)
#   - Extra: an overlay showing agreement -- cortex in the red channel, manual
#     in the green channel; overlap appears yellow:
#         overlay = np.zeros((*img.shape, 3))
#         overlay[..., 0] = cortex
#         overlay[..., 1] = manual_mask

# --- your code here ---


# ===========================================================================
# IF YOU FINISH EARLY
# ===========================================================================
# 1. Look at the overlay. Where does the automatic method disagree with your
#    manual segmentation? Are those places where the cortex boundary is unclear?
# 2. Try your script on a Nissl annotation AND a myelin annotation. Does the
#    intensity flip in Step 1 let the same code handle both?
# 3. What happens to Step 3 on a section where the two hemispheres are joined
#    into one blob? What would you change?
