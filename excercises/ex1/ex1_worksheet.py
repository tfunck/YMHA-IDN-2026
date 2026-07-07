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

# --- your file paths here ---
# IMAGE_PATH = "section.jpg"      # change to your own image file


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

# --- your code here ---


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

# --- your code here ---


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

# --- your code here ---


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

# --- your code here ---


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

# --- your code here ---


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

# --- your code here ---


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
#   - os.path.basename("folder/section.jpg") -> "section.jpg" (drops the folder)
#   - os.path.splitext("section.jpg") -> ("section", ".jpg")  (splits extension)
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

# --- your code here ---


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
