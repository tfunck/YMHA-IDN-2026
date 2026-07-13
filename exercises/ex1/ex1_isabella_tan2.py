"""
First image-segmentation exercise -- WORKSHEET
==============================================

Welcome! In this exercise you'll take a brain-section image and separate the
tissue from the background in a few different ways. You'll build the script up
one step at a time.

How to work through this:
  - Each step below is a task with hints, not finished code.
  - Write your code in the blank space under each step.
  - Run the script often (after every step) so you catch mistakes early:

        python segmentation_exercise_worksheet.py

  - If you get truly stuck on a step, a full solution exists in
    `segmentation_exercise.py` -- but try the hints first! You'll learn far
    more from a few minutes of being stuck than from copying the answer.

Getting help from the hints:
  - Every library function mentioned below has documentation. Search the web for
    e.g. "imageio imread" or "matplotlib subplot" to see examples of its use.
  - When something breaks, READ THE ERROR MESSAGE. The last line usually says
    what went wrong; the file/line number says where.
"""

# ===========================================================================
# SETUP - imports and file paths
# ===========================================================================
# You'll need four tools in this exercise. Add import statements for:
#   - imageio's v3 module, to read the image     (hint: import imageio.v3 as iio)
#   - numpy, for array math                      (hint: the usual alias is np)
#   - matplotlib's pyplot, to plot and save      (hint: the usual alias is plt)
#   - Otsu's threshold from scikit-image         (hint: it lives in skimage.filters
#                                                 and is called threshold_otsu)
#
# Also define, near the top, the path to the image you want to open and names
# for the output files you'll save. Keeping paths in named variables at the top
# (instead of typed inline later) makes them easy to change.

# --- your imports here ---
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio
import glob
import os

# --- your file paths here ---
IMAGE_PATH = "catsec_m117.jpg"
COMPARISON_PATH = "comparison.png"
HISTOGRAM_PATH = "histogram.png"
OTSU_OUT = "otsu_mask.png"

# ===========================================================================
# STEP 1 - Open the image
# ===========================================================================
# TASK: Load the image file into a variable called `img`.
#
# Hints:
#   - The function is  iio.imread(...).  It takes the file path.
#   - Add the argument  mode="L"  to load the image in GRAYSCALE (one brightness
#     value per pixel). Without it, a colour photo loads as three colour
#     channels, which makes "brightness" thresholding confusing later.
#   - After loading, PRINT some facts about what you got: img.shape (the size),
#     img.dtype (the number type), img.min() and img.max() (darkest/brightest
#     pixel). Inspecting your data before using it will save you hours later.
#
# Think about: what does img.shape look like with mode="L" versus without it?
# What do you think the extra number means?

img = iio.imread("catsec_m117.jpg", mode="L")
print(img.shape)
print(img.dtype)
print(img.min())
print(img.max())

# without mode=L the scan becomes colorful

# ===========================================================================
# STEP 2 - Segment with a threshold you choose by hand
# ===========================================================================
# TASK: Make a black-and-white "mask" that is True where the image is brighter
# than a cut-off and False elsewhere. Call it `segmented_mask`.
#
# Hints:
#   - First pick a cut-off number, `threshold`. A reasonable starting guess is
#     halfway up the brightness range: img.max() * 0.5.
#   - Then compare the WHOLE image to it in one line:  img > threshold.
#     This runs on every pixel at once (the point of NumPy arrays) and returns
#     an array of True/False the same size as the image.
#   - Print the threshold value so you know what you used.
#
# Play: once it runs, try 0.3 and 0.7 instead of 0.5. How does the mask change?

threshold = img.max() * 0.5
segmented_mask = img > threshold
print("Manual threshold used:", threshold)

# 255 * 0.5 = 127.5
# the dark areas of the mask increase as the threshold decreases (0.7) while
# the light areas increase as the threshold increases (0.3). the mask is more
# sensitive to the threshold value, and the mask changes significantly with
# different thresholds.
# note: for "segmented_mask = img>threshold" if a pixel is greater than the threshold,
# it becomes True (white) in the mask otherwise it becomes False (black), compares it
# to 127.5, making it an array of true or false --- black or white for the mask, hence
# the img>threshold
# for print("Manual threshold used", threshold) it prints the words and then the threshold
# number which when it is 0.5, it is 127.5


# ===========================================================================
# STEP 3 - Plot the image and the mask side by side, and save to a .png
# ===========================================================================
# TASK: Make one figure with two panels -- original on the left, your mask on
# the right -- and SAVE it as a .png.
#
# Hints:
#   - Start a figure:  plt.figure(figsize=(8, 4)).
#   - plt.subplot(1, 2, 1) splits the figure into 1 row x 2 columns and selects
#     the FIRST panel; then plt.imshow(...) draws into it.
#   - plt.subplot(1, 2, 2) selects the SECOND panel; draw the mask there.
#   - IMPORTANT: pass  cmap="gray"  to imshow, e.g. plt.imshow(img, cmap="gray").
#     Without it, matplotlib colours a grayscale image blue-green-yellow and
#     your brain will look radioactive.
#   - Nice-to-haves: plt.title("...") to label each panel, plt.axis("off") to
#     hide the number axes.
#   - Save with  plt.savefig(SOME_PATH).  Do NOT call plt.show() before it --
#     show() clears the figure, so a savefig after it saves a blank image.
#
# When it works, open the saved .png and check it looks right.

# makes a new figure/canvas to draw on at 8 inches wide and 4 inches tall
plt.figure(figsize=(8, 4))

# tells matplotlib to create 1 row and 2 columns like a grid
plt.subplot(1, 2, 1)
# imshow = image show and displays the image in gray
plt.imshow(img, cmap="gray")
plt.title("Original image")  # gives it a title for original
plt.axis("off")  # turns off axis so it doesn't show the x and y axis

plt.subplot(1, 2, 2)
plt.imshow(segmented_mask, cmap="gray")
plt.title("Manual threshold mask")
plt.axis("off")

plt.tight_layout()  # adjusts format and spacing of the figure so that things don't overlap
plt.savefig(COMPARISON_PATH)  # saves figure to a file called comparison_path
plt.close()  # closes the figure

# how do we make sure they're side by side? => what matlib is for

print("Saved comparison image to:", COMPARISON_PATH)


# ===========================================================================
# STEP 4 - Plot the histogram of the image
# ===========================================================================
# TASK: Draw a histogram of the image's pixel brightnesses and save it.
#
# What's a histogram here? It counts how many pixels have each brightness. A
# brain section on a bright background often shows two humps -- a tall one for
# the bright background and another for the darker tissue. A good threshold sits
# in the VALLEY between the humps.
#
# Hints:
#   - The function is  plt.hist(values, bins=256).
#   - plt.hist wants a flat 1-D list of numbers, but your image is 2-D. Flatten
#     it first with  img.ravel().
#   - Label your axes: plt.xlabel(...), plt.ylabel(...), plt.title(...).
#   - Start a fresh plt.figure() first so it doesn't draw on the previous plot,
#     then save to its own .png.
#
# Look at your histogram: can you spot the two humps? Where's the valley?
# creates a new canvas of 6 inches wide and 4 inches tall
plt.figure(figsize=(6, 4))

plt.hist(img.ravel(), bins=256)
# img. rave turns image from 2D gird into list of pixel
# brightness values and bins makes one bar for each value (theres 256), this is what
# creates the histogram/is the histogram function

plt.xlabel("Pixel brightness")  # labels x axis pixel brightness
plt.ylabel("Number of pixels")  # labels y axis number of pixels
plt.title("Histogram")  # gives it histogram title
plt.tight_layout()  # formatting
plt.savefig(HISTOGRAM_PATH)  # saves it under histogram png
print("Saved histogram to:", HISTOGRAM_PATH)


# ===========================================================================
# STEP 5 - Compute the Otsu threshold automatically and print it
# ===========================================================================
# TASK: Let Otsu's method choose the threshold for you. Store it in
# `otsu_threshold` and print it.
#
# What's Otsu? A classic method that reads the histogram and automatically finds
# the cut-off that best separates the two humps. One function call.
#
# Hints:
#   - You imported it in SETUP:  threshold_otsu.
#   - It takes the image and returns a single number:
#       otsu_threshold = threshold_otsu(img)
#   - Print it, then compare it to your hand-picked threshold from Step 2 and to
#     the valley you eyeballed in the Step 4 histogram. Are they close?

otsu_threshold = threshold_otsu(img)
# uses original image(img)'s numpy array to create its own histogram from those pixels
# chooses its own threshold based off this histogram and stores the answer in otsu_threshold
print(threshold_otsu(img))
# prints this number in the terminal (190)

# ===========================================================================
# STEP 6 - Apply the Otsu threshold to make a mask and plot it
# ===========================================================================
# TASK: Build a mask using the Otsu threshold (same idea as Step 2, new number),
# then make a figure comparing the original, your MANUAL mask, and the OTSU
# mask, and save it.
#
# Hints:
#   - The mask is just  img > otsu_threshold.
#   - This time make THREE panels: plt.subplot(1, 3, 1), (1, 3, 2), (1, 3, 3).
#     Original first, manual mask second, Otsu mask third.
#   - Remember cmap="gray" on every imshow.
#   - Give each panel a title. You can put the number in with an f-string, e.g.
#       plt.title(f"Otsu (>{otsu_threshold:.0f})")
#   - Save to a .png and open it.
#
# The payoff: does the Otsu mask look better or worse than your best hand-picked
# one? Look back at the Step 4 histogram -- can you see WHY Otsu chose its value?
# The otsu mask looks better than my manual threshold becuase the manual threshold
# only got the brightest parts while the otsu mask looks very close to the original
# image. otsu chose a threshold of 190 because it automatically found the best
# threshold based on the bright background and dark brain tissue and based on the histogram.

otsu_mask = img > otsu_threshold
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(segmented_mask, cmap="gray")
plt.title(f"Manual (>{threshold:.0f})")
# shows otsu threshold in the title based off the threshold (don't really understand the
# exact syntax of this line though like what each part does)
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(otsu_mask, cmap="gray")
plt.title(f"Otsu (>{otsu_threshold:.0f})")
plt.axis("off")

plt.savefig(OTSU_OUT, dpi=150)
# dpi 150 makes the image higher quality
print("Saved OTSU comparison to:", OTSU_OUT)


# ===========================================================================
# STEP 7 - Process a WHOLE FOLDER of images (function + loop)
# ===========================================================================
# Your script handles ONE image. Real datasets have hundreds. This step teaches
# the two tools that scale it up: wrapping your code in a FUNCTION, and using
# glob to LOOP over many files. This is the trickiest step -- take it slowly.
#
# TASK, part A: wrap the "process one image" work into a function.
#   - Define a function that takes an input path and an output path:
#         def segment_one(input_path, output_path):
#             ...
#   - Move your load -> Otsu threshold -> mask -> save code INSIDE it, but use
#     the function's arguments instead of hard-coded paths. So
#     `iio.imread(input_path, mode="L")` instead of a fixed filename, and
#     `plt.savefig(output_path)` instead of a fixed output name.
#   - Everything the function needs should come in through its two arguments.
#
# TASK, part B: call it for every image in a folder.
#   - At the top of the file, add:  import glob   and   import os
#   - glob.glob("*.jpg") returns a LIST of every filename ending in .jpg in the
#     current folder. The "*" means "match anything".
#   - Loop over that list and call your function once per file:
#         for path in glob.glob("*.jpg"):
#             ...
#             segment_one(path, output_name)
#
# Hints for building the output name (so each image gets its own output file):
#   - You don't want every image to save over the same output file! Build a
#     unique output name from each input name.
#   - os.path.basename("folder/") -> (drops the folder)
#   - os.path.splitext -> ("section", ".jpg")  (splits extension)
#   - So:  stem = os.path.splitext(os.path.basename(path))[0]   gives "section",
#     and then  output_name = stem + "_segmented.png".
#   - Print something inside the loop (e.g. which file you just did) so you can
#     watch progress and see it's working.
#
# Debugging tips (you'll probably need them here):
#   - Get the function working on ONE file first (call it once by hand), THEN
#     add the loop. Don't write both at once.
#   - If glob finds nothing, print glob.glob("*.jpg") to see what it matched --
#     you may be in the wrong folder, or the files may end in .JPG or .png.
#   - A crash on one image shouldn't be mysterious: read the error, note WHICH
#     file it was on (that's why printing inside the loop helps).

def segment_one(input_path, output_path):
    im = iio.imread(input_path, mode="L")
    t = threshold_otsu(im)
    mask = im > t

    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(im, cmap="gray")
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(otsu_mask, cmap="gray")
    plt.title(f"Otsu (>{otsu_threshold:.0f})")

    plt.savefig(output_path, dpi=150)


input_files = glob.glob("*.jpg")
print("Found files:", input_files)
for path in input_files:
    stem = os.path.splitext(os.path.basename(path))[0]
    output_name = stem + "_segmented.png"
    segment_one(path, output_name)
    print(f"Processed {path} -> {output_name}")


# ===========================================================================
# IF YOU FINISH EARLY - things to explore
# ===========================================================================
# 1. Load the SAME image WITHOUT mode="L" and print img.shape. What changed, and
#    what does the extra dimension mean?
# 2. In Step 2, loop over several thresholds (0.3, 0.4, 0.5, 0.6, 0.7) and save a
#    mask for each. Which fraction works best for your image?
# 3. The background here is bright and the tissue is darker. What would you
#    change so the MASK marks the TISSUE (dark) instead of the background?
#    (Hint: which way does the > point?)
# 4. Try your script on a different species' section. Does the same threshold
#    still work? Why might it not?
