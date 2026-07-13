"""
First image-segmentation exercise
==================================

Goal: open a brain-section image, separate the tissue from the background a
couple of different ways, and look at the results.

By the end you will have:
  1. loaded an image,
  2. segmented it with a threshold you pick by hand,
  3. saved a picture comparing the original and the mask,
  4. drawn the image's histogram,
  5. computed the Otsu threshold automatically and applied it.

Run it from the terminal like this:

    python segmentation_exercise.py

Before running, change IMAGE_PATH below to point at one of your own sections.
"""

# ---------------------------------------------------------------------------
# Imports: these are the toolboxes we borrow code from.
# ---------------------------------------------------------------------------
import os              # helps build output filenames from input filenames
# finds files matching a pattern (put with imports up top)
import glob
import imageio.v3 as iio          # reading image files into arrays
import numpy as np                # working with arrays of numbers (the image)
import matplotlib.pyplot as plt   # making and saving plots
from skimage.filters import threshold_otsu   # the automatic Otsu threshold

# ---------------------------------------------------------------------------
# Settings: the two file paths. Change these to your own files.
# Keeping paths in one place at the top makes them easy to find and edit.
# ---------------------------------------------------------------------------
# <-- put your image filename here
IMAGE_PATH = "exercises/ex1/catsec_m117.jpg"
# where the side-by-side image is saved
COMPARISON_PATH = "comparison.png"
HISTOGRAM_PATH = "histogram.png"           # where the histogram is saved
OTSU_PATH = "otsu_mask.png"                # where the Otsu result is saved


# ===========================================================================
# STEP 1 - Open the image
# ===========================================================================
# imageio reads the file from disk and hands us back a NumPy array: a grid of
# numbers where each number is how bright one pixel is. We pass mode="L" to get
# a single-channel GRAYSCALE image ("L" = luminance). Without it, a colour JPG
# comes back with 3 channels (red, green, blue), which makes thresholding on
# "brightness" ambiguous. Grayscale keeps this first exercise simple.
img = iio.imread(IMAGE_PATH, mode="L")

# It is always worth checking what you actually loaded. .shape tells you the
# size (height, width); .dtype tells you the number type; .min()/.max() tell
# you the darkest and brightest pixel values.
print("Image loaded.")
print("  shape (height, width):", img.shape)
print("  pixel value type     :", img.dtype)
print("  darkest / brightest  :", img.min(), "/", img.max())


# ===========================================================================
# STEP 2 - Segment with a threshold you choose by hand
# ===========================================================================
# "Segmenting" here just means deciding, for every pixel, whether it is
# foreground or background. The simplest rule: pick a brightness cut-off and
# say every pixel brighter than it is foreground.
#
# img.max() * 0.5 means "halfway between zero and the brightest pixel".
# Try changing 0.5 to 0.3 or 0.7 and re-running to see how the mask changes.
threshold = img.max() * 0.5

# The comparison "img > threshold" is applied to EVERY pixel at once (this is
# the magic of NumPy arrays). The result, segmented_mask, is an array of the
# same shape containing True/False: True where the pixel is above the cut-off.
segmented_mask = img > threshold

print("\nManual threshold used:", threshold)


# ===========================================================================
# STEP 3 - Plot the image and the mask side by side, and save it
# ===========================================================================
# plt.subplot(rows, cols, n) carves the figure into a grid and selects the
# n-th cell to draw in. Here we make 1 row x 2 columns.
plt.figure(figsize=(8, 4))          # start a new figure, 8x4 inches

plt.subplot(1, 2, 1)                # first cell (left)
# show the original; cmap="gray" = greyscale
plt.imshow(img, cmap="gray")
plt.title("Original")               # a label above the picture
# hide the x/y number axes (not useful here)
plt.axis("off")

plt.subplot(1, 2, 2)                # second cell (right)
plt.imshow(segmented_mask, cmap="gray")
plt.title("Manual threshold mask")
plt.axis("off")

plt.tight_layout()                  # tidy up spacing so titles don't overlap
plt.savefig(COMPARISON_PATH, dpi=150)   # SAVE to a file (do this before show)
plt.close()                         # close the figure to free memory
print("Saved comparison image to:", COMPARISON_PATH)


# ===========================================================================
# STEP 4 - Plot the histogram of the image
# ===========================================================================
# A histogram counts how many pixels fall into each brightness level. It shows
# you the "shape" of the image's brightness. For a brain section on a bright
# background you often see two humps: a tall one for the bright background and
# another for the darker tissue. The threshold's job is to sit in the valley
# between those humps.
#
# .ravel() flattens the 2-D image into one long 1-D list of pixel values,
# which is the form plt.hist expects.
plt.figure(figsize=(6, 4))
plt.hist(img.ravel(), bins=256, color="gray")
plt.title("Pixel brightness histogram")
plt.xlabel("Brightness (0 = black, 255 = white)")
plt.ylabel("Number of pixels")
plt.tight_layout()
plt.savefig(HISTOGRAM_PATH, dpi=150)
plt.close()
print("Saved histogram to:", HISTOGRAM_PATH)


# ===========================================================================
# STEP 5 - Calculate the Otsu threshold automatically and print it
# ===========================================================================
# Choosing the threshold by hand is fiddly. Otsu's method looks at the
# histogram and automatically finds the cut-off that best separates the two
# humps (formally: it minimises the spread within each group). scikit-image
# gives it to us in one line.
otsu_threshold = threshold_otsu(img)
print("\nOtsu threshold (chosen automatically):", otsu_threshold)


# ===========================================================================
# STEP 6 - Apply the Otsu threshold to make a mask and plot it
# ===========================================================================
# Exactly the same idea as Step 2, but using Otsu's value instead of our
# hand-picked one.
otsu_mask = img > otsu_threshold

# Draw the original, the manual mask, and the Otsu mask together so you can
# compare how the two thresholds did.
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(segmented_mask, cmap="gray")
plt.title(f"Manual (>{threshold:.0f})")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(otsu_mask, cmap="gray")
plt.title(f"Otsu (>{otsu_threshold:.0f})")
plt.axis("off")

plt.tight_layout()
plt.savefig(OTSU_PATH, dpi=150)
plt.close()
print("Saved Otsu comparison to:", OTSU_PATH)

print("\nSteps 1-6 done! Open the three .png files to see your results.")


# ===========================================================================
# STEP 7 - Process a WHOLE FOLDER of images with a function and a loop
# ===========================================================================
# Right now the script handles ONE image. Real datasets have hundreds. Two ideas
# make that easy:
#   (a) wrap all the work for one image into a FUNCTION you can call repeatedly;
#   (b) use glob to get a LIST of file paths, then LOOP over them.


def segment_one(input_path, output_path):
    """Do the whole pipeline for a SINGLE image: load it, compute the Otsu
    threshold, make a mask, and save a side-by-side figure.

    Parameters
    ----------
    input_path : str   path to the image to read
    output_path : str  path to the .png to write
    """
    # --- load (grayscale) ---
    im = iio.imread(input_path, mode="L")

    # --- Otsu threshold + mask ---
    t = threshold_otsu(im)
    mask = im > t

    # --- save a figure ---
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap="gray")
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap="gray")
    plt.title(f"Otsu mask (>{t:.0f})")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"  processed {input_path} -> {output_path}")


# Find every .jpg in the current folder. The "*" means "anything", so "*.jpg"
# matches all files ending in .jpg. Change the pattern/folder to suit your data.
input_files = glob.glob("*.jpg")
print(f"\nFound {len(input_files)} image(s) to process:", input_files)

# Loop over them, building an output name from each input name.
for path in input_files:
    # turn "section.jpg" into "section_segmented.png"
    stem = os.path.splitext(os.path.basename(path))[0]   # "section"
    out = stem + "_segmented.png"
    segment_one(path, out)

print("\nAll images processed!")
