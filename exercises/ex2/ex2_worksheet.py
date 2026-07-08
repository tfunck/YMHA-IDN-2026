"""
Cortex isolation + Dice evaluation -- WORKSHEET
===============================================

These myelin sections have a BRIGHT background, PALE cortex, and DARK white
matter. Your job: isolate the cortex, then measure how good it is against YOUR
manual segmentation using the Dice score.

You will provide TWO files:
  - the raw section image, and
  - your own manual segmentation of the cortex (white = cortex on black).

Work through the steps in order, running often:

    python cortex_pipeline_worksheet.py

If you get stuck, a full solution is in `cortex_pipeline.py` -- but try the
hints first. When something breaks, READ THE ERROR (last line = what, the line
number = where).
"""

# ---------------------------------------------------------------------------
# SETUP - add the imports you'll need and set your file paths.
#   - imageio.v3 (as iio), numpy (as np), matplotlib.pyplot (as plt)
#   - from scipy import ndimage as ndi           (connected components)
#   - from skimage.filters import threshold_otsu (automatic threshold)
# ---------------------------------------------------------------------------

# --- your imports here ---

# IMAGE_PATH = "redroo.jpg"            # your raw section
# MANUAL_PATH = "redroo_manual.png"    # your manual cortex segmentation
# OUTPUT_PATH = "cortex_pipeline.png"

# Load the raw image in grayscale (hint: iio.imread(IMAGE_PATH, mode="L")).
# --- your code here ---


# ===========================================================================
# STEP 1 - Flip the image intensities
# ===========================================================================
# The background is bright (~255) and tissue is darker. Several later steps are
# easier when the BACKGROUND is dark, so invert the image: value v -> 255 - v.
#
# Hint: with a NumPy array this is a one-liner:  flipped = 255 - img

# --- your code here ---


# ===========================================================================
# STEP 2 - Isolate the foreground by thresholding away the background
# ===========================================================================
# TASK: make a True/False mask `foreground` that is True on tissue, False on
# background.
#
# Hints:
#   - The background fills most of the frame, so the MOST COMMON pixel value is
#     a good estimate of the background brightness. Find it with:
#         values, counts = np.unique(img, return_counts=True)
#         background_value = values[np.argmax(counts)]
#   - In the FLIPPED image the background is near (255 - background_value), and
#     tissue is brighter than that. So threshold the flipped image a little
#     above the flipped background:
#         foreground = flipped > (255 - background_value + 15)
#     The "+ 15" is a small margin so you don't catch background noise. Try
#     changing it and see what happens.

# --- your code here ---


# ===========================================================================
# STEP 3 - Keep only the two largest connected components (the hemispheres)
# ===========================================================================
# TASK: the foreground still has junk (text, dust). Keep only the two biggest
# connected blobs -- the two hemispheres -- in a mask called `tissue`.
#
# Hints:
#   - ndi.label(foreground) returns (labels, num_features): an image where each
#     separate blob has a unique integer id, and how many blobs there are.
#   - Measure each blob's size with:
#         sizes = ndi.sum(np.ones_like(labels), labels, range(1, num_features + 1))
#   - The two biggest ids are:
#         two_largest_ids = np.argsort(sizes)[::-1][:2] + 1
#     (argsort sorts small->large, [::-1] reverses it, [:2] takes the first two;
#      the +1 is because blob ids start at 1 but sizes is 0-indexed.)
#   - Build the mask with:  tissue = np.isin(labels, two_largest_ids)
#
# Think: why TWO components? On a section where the hemispheres are joined,
# what would you change?

# --- your code here ---


# ===========================================================================
# STEP 4 - Otsu threshold inside the tissue to isolate cortex
# ===========================================================================
# TASK: split the tissue into cortex vs white matter with Otsu, keeping cortex
# in a mask called `cortex_otsu`.
#
# Hints:
#   - Compute the threshold using ONLY the tissue pixels:
#         otsu_threshold = threshold_otsu(flipped[tissue])
#   - CAREFUL about direction: because you FLIPPED the image, the originally
#     pale cortex is now the DARKER class. So cortex is the tissue pixels BELOW
#     the threshold:
#         cortex_otsu = tissue & (flipped < otsu_threshold)

# --- your code here ---


# ===========================================================================
# STEP 5 - Evaluate with the Dice score
# ===========================================================================
# The Dice score measures how well a predicted mask matches a reference mask.
# It's 0 for no overlap and 1 for a perfect match:
#
#     Dice = 2 * (pixels in BOTH masks) / (pixels in pred + pixels in truth)
#
# TASK: load your manual segmentation, finish the dice_score function below, and
# print the Dice of the Otsu mask vs the manual one.
#
# Hints:
#   - Load the manual mask as grayscale and make it boolean (white = cortex):
#         manual = iio.imread(MANUAL_PATH, mode="L")
#         manual_mask = manual > 0
#   - Fill in the two lines marked TODO in the function.
#   - "pixels in both" is a logical AND of the two masks, then .sum().
#   - "pixels in a mask" is just that mask's .sum() (True counts as 1).

def dice_score(pred, truth):
    """Return the Dice score between two boolean masks (0 = none, 1 = perfect)."""
    pred = pred.astype(bool)
    truth = truth.astype(bool)

    # TODO: count the pixels that are True in BOTH masks.
    #       hint: np.logical_and(pred, truth).sum()
    overlap = 0   # <-- replace this

    # TODO: return  2 * overlap / (number of True in pred + number of True in truth)
    #       hint: pred.sum() and truth.sum() give those counts
    return 0.0    # <-- replace this

# --- load the manual mask and print the Dice score here ---
# dice_otsu = dice_score(cortex_otsu, manual_mask)
# print("Dice (Otsu vs manual):", dice_otsu)


# ===========================================================================
# STEP 6 - Make an output figure
# ===========================================================================
# TASK: save one figure showing the original, each step's result, and the Dice
# score (put it in the cortex panel's title).
#
# Hints:
#   - fig, ax = plt.subplots(2, 3, figsize=(14, 9)); ax = ax.ravel()
#   - Draw into ax[0], ax[1], ... with imshow, e.g.
#         ax[0].imshow(img, cmap="gray"); ax[0].set_title("Original")
#   - Put the Dice score in a title with an f-string:
#         ax[4].set_title(f"Cortex - Otsu  Dice={dice_otsu:.3f}")
#   - Turn axes off: for a in ax: a.axis("off")
#   - Save: plt.savefig(OUTPUT_PATH, dpi=150)
#
# Extra: try an OVERLAY panel that shows agreement. Put the Otsu mask in the red
# channel and the manual mask in the green channel of an RGB image -- where they
# overlap you'll see yellow:
#     overlay = np.zeros((*img.shape, 3))
#     overlay[..., 0] = cortex_otsu
#     overlay[..., 1] = manual_mask

# --- your code here ---


# ===========================================================================
# IF YOU FINISH EARLY
# ===========================================================================
# 1. The cortex mask probably also grabbed some pale non-cortex structures.
#    Look at the overlay -- where does Otsu disagree with your manual mask?
# 2. Try changing the "+ 15" margin in Step 2. How sensitive is the result?
# 3. Run the whole thing on a different section. Does the same pipeline still
#    work, or does one step need adjusting?
