Exercise 3: Evaluating Segmentations Across a Whole Dataset
==========================================================

In Exercise 2 you segmented the cortex from one section and scored it against
one manual annotation. Now you'll scale that up: run the evaluation across a
whole folder of annotations at once, and summarise how well the automatic method
does for each species.

By the end you'll have a script you run from the command line like this:

    python batch_dice_eval.py ANNOTATION_FOLDER sect_info.csv

...that finds every annotation, scores it, and prints the average Dice score per
species.

Work through the steps in order. Each is a task with hints — try the hints
before asking for help. When something breaks, READ THE ERROR (the last line
says what went wrong; the line number says where).


What you're given
-----------------

- A folder containing annotation files. Each one is named `<something>_manual.tif`
  and is a two-layer TIFF (top layer = manual segmentation, bottom layer = raw
  image), exactly like Exercise 2.
- `sect_info.csv`, a table describing every section. Its columns are:
  `specimen`, `species`, `stain`, `filename`, `section`. The `filename` column
  holds a path like `sections/cat_nissl/catsec_m105.jpg`.

You'll connect the two: for each `_manual.tif` file, look up its row in the CSV
to find out the species and stain, run the segmentation, and record the Dice
score.


Step 1 — Find all the annotation files
--------------------------------------

**Task:** get a list of every `*_manual.tif` file inside a folder (including
sub-folders).

**Hints:**
- The `glob` module finds files matching a pattern. The pattern `*_manual.tif`
  matches your annotation files.
- To search sub-folders too, use the `**` wildcard and `recursive=True`:

      import glob, os
      pattern = os.path.join(annotation_dir, "**", "*_manual.tif")
      files = glob.glob(pattern, recursive=True)

- Print how many you found. If it's zero, print the pattern and check you're
  pointing at the right folder.


Step 2 — Get the base filename for matching
-------------------------------------------

The paths in the CSV (`sections/cat_nissl/catsec_m105.jpg`) won't match where
the files actually live on your computer. So you match on the **base filename
without its extension** instead — the `catsec_m105` part.

**Task:** for a given file path, extract just that stem.

**Hints:**
- `os.path.basename("a/b/catsec_m105.jpg")` → `"catsec_m105.jpg"` (drops folders).
- `os.path.splitext("catsec_m105.jpg")` → `("catsec_m105", ".jpg")` (splits ext).
- Your annotation files end in `_manual.tif`, so their stem is
  `catsec_m105_manual`. You'll want to strip the `_manual` part too, so the stem
  matches the CSV's `catsec_m105`. Think about how to remove that suffix (hint:
  string slicing, or `.replace("_manual", "")`).


Step 3 — Build a lookup from the CSV
------------------------------------

**Task:** load `sect_info.csv` and make it easy to look up a row by its stem.

**Hints:**
- Load it with pandas: `df = pd.read_csv(csv_path)`.
- Make a new `stem` column by applying the Step-2 logic to the `filename`
  column:

      df["stem"] = df["filename"].apply(
          lambda f: os.path.splitext(os.path.basename(f))[0]
      )

- Now you can find the row(s) for a stem with `df[df["stem"] == my_stem]`.

**Watch out — a subtlety worth knowing:** some stems appear **twice** in the CSV,
once as a nissl section and once as a myelin section (the same basename lives in
both a `_nissl` and a `_myelin` folder). So a lookup might return *two* rows. For
now, notice when this happens and print a warning; think about how you might pick
the right one (a hint: does the annotation file's folder path contain the word
"nissl" or "myelin"?). Handling this cleanly is a bonus — get the simple case
working first.


Step 4 — Reuse your segmentation and Dice code
----------------------------------------------

**Task:** bring in the functions you wrote in Exercise 2 — `read_annotation`,
the cortex segmentation, and `dice_score` — so you can call them here.

**Hints:**
- You already wrote and tested these. Copy them in as functions at the top of
  the script (or, more elegantly, import them from your Exercise 2 file).
- Remember the **stain matters**: Nissl grey matter is dark, myelin grey matter
  is bright. Your segmentation should flip the image based on the `stain` value
  from the CSV row, so grey matter ends up as the same class either way. Pass the
  stain into your segmentation function.


Step 5 — Add CLAHE local contrast enhancement
---------------------------------------------

CLAHE (Contrast Limited Adaptive Histogram Equalization) boosts *local* contrast,
which can make the grey-matter boundary easier to threshold, especially where
staining is uneven across the section.

**Task:** apply CLAHE to the raw image before you threshold it.

**Hints:**
- It lives in scikit-image: `from skimage.exposure import equalize_adapthist`.
- It returns a float image scaled 0–1, so multiply by 255 if the rest of your
  code expects 0–255 values:

      enhanced = equalize_adapthist(raw, clip_limit=0.01) * 255.0

- `clip_limit` controls how aggressive it is. Try a few values (e.g. 0.005,
  0.01, 0.02) and see how the segmentation changes.


Step 6 — Loop over every file and collect results
-------------------------------------------------

**Task:** for each annotation file: find its CSV row, read the two layers,
segment, score, and store the result.

**Hints:**
- Build a list of dictionaries, one per file, then turn it into a DataFrame at
  the end:

      records = []
      for path in files:
          ...
          records.append({
              "stem": stem, "species": species, "stain": stain, "dice": d
          })
      results = pd.DataFrame(records)

- Wrap the per-file work in a `try/except` so one broken file doesn't stop the
  whole run — record a `dice` of `NaN` (use `float("nan")`) and a note instead.
- Print each result as you go (`stem`, species, Dice) so you can watch progress
  and spot problems.
- What should happen if a stem has **no** matching CSV row? Decide, and handle
  it (e.g. record it with a note and move on).


Step 7 — Summarise average Dice by species
-------------------------------------------

**Task:** print the mean Dice score for each species.

**Hints:**
- pandas `groupby` does this in one line:

      by_species = results.groupby("species")["dice"].mean()
      print(by_species.sort_values(ascending=False))

- Showing the **count** per species alongside the mean is more informative (a
  mean over 2 sections is less trustworthy than over 50):

      results.groupby("species")["dice"].agg(["mean", "count"])

- Remember some Dice values may be `NaN` (broken/unmatched files). pandas skips
  `NaN` in `mean()` by default, but check the counts so you know how many
  actually contributed.


Step 8 — Save the results
-------------------------

**Task:** write the full per-file table to a CSV so you can look at it later.

**Hints:**
- `results.to_csv("dice_results.csv", index=False)`.
- Open it in a spreadsheet and sort by Dice. Which sections scored worst? Look
  at a couple of them — is the segmentation genuinely bad there, or is something
  else going on (wrong stain, ambiguous stem, unclear cortex boundary)?


If you finish early
-------------------

1. **Nissl vs myelin.** Group your results by `stain` instead of species. Does
   the automatic method do better on one stain type than the other? Why might
   that be?
2. **CLAHE on/off.** Run the whole thing with and without the CLAHE step. Does it
   actually improve the average Dice, or only on some species?
3. **The ambiguous stems.** Go back to the duplicate-basename problem from Step 3
   and make your script pick the correct row using the folder name. Confirm it
   now assigns the right stain to those files.
4. **Worst performers.** Make a plot (or just a sorted table) of Dice by species.
   The species with the lowest scores are the ones where simple thresholding
   struggles most — those are exactly the cases the later parts of the project
   (better features, neural networks) are meant to improve.
