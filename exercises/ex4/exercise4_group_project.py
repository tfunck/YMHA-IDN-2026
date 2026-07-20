Exercise 4 (Group Project): A Complete Segmentation Pipeline
===========================================================

This is a **team exercise**. Together you'll build a *single* shared script that
segments every available brain section, checks how well it works against the
manual annotations, and then produces segmentations for all the sections that
nobody has hand-labelled yet.

This is the closest thing yet to a real research tool: it has to handle the whole
dataset, cope with the two different stain types, be honest about its own
accuracy, and produce output in a standard format. Because it's one script built
by several people, you'll also need to agree on how to work together.


The three jobs your script must do
----------------------------------

1. **Segment** every section using CLAHE + Otsu, with the best configuration you
   found in Exercise 3 — and correctly handling the contrast difference between
   Nissl and myelin sections.
2. **Validate** the automatic segmentations against the manual ones, so you know
   how much to trust the method.
3. **Apply** the method to the sections that don't have a manual label yet, and
   save each result in the same two-layer TIFF format as the `_manual` files.


What you're given
-----------------

- `sect_info_with_manual.csv` — the section table. It has the usual columns
  (`specimen`, `species`, `stain`, `filename`, `section`) plus a **`manual`**
  column. Rows where `manual` is filled in have a hand-drawn label you can
  validate against; rows where it's empty are the ones still needing
  segmentation.
- The section images themselves, and the `_manual_label` files referenced in the
  `manual` column.


Working together
----------------

Before writing code, agree as a team on a few things — this matters more than it
sounds, because a shared script falls apart fast without it:

- **Split the work by job, not by person editing the same lines.** A natural
  division is: one person on the segmentation function (job 1), one on the
  validation loop (job 2), one on the apply-and-save loop (job 3), and everyone
  agreeing on the shared pieces below.
- **Agree on the shared functions first.** The three jobs all need to load an
  image, segment it, and read/write annotation TIFFs. Define those function
  *names and what they take/return* as a team before anyone writes the insides,
  so your pieces fit together. For example, decide now that everyone will use:

      segment_section(raw_image, stain, ...)   -> boolean cortex mask
      read_annotation(tif_path)                -> (seg_layer, raw_layer)
      write_annotation(path, seg_mask, raw)    -> saves a two-layer TIFF

- **Use Git and small commits.** Pull before you start, commit small working
  pieces often, and write clear commit messages. If two people edit the same
  function, talk first. (This is exactly what version control is for.)
- **Test on a few sections before running on all ~1700.** Get it working end to
  end on maybe five sections spanning both stains, then scale up.


Job 1 — Segment every section (handling Nissl vs myelin)
--------------------------------------------------------

**Task:** write a `segment_section` function that takes a raw image and its
stain, and returns a cortex mask — using your best CLAHE + Otsu settings.

**The key challenge — stain contrast.** Remember from Exercise 2:
- In **Nissl** sections, grey matter is **dark**.
- In **myelin** sections, grey matter is **bright**.

If you apply the same Otsu rule to both, you'll segment grey matter in one and
white matter in the other. Your function needs to account for this. Think about
where the `stain` value comes in:
- One clean approach: **flip** the myelin images (`img.max() - img`) so that grey
  matter is *always* the dark class, then use one consistent rule ("keep below
  the Otsu threshold") for everything.
- The `stain` column in the CSV tells you which sections need flipping. Pass the
  stain into `segment_section` and branch on it.
- Don't forget the sections labelled `nissl+myelin` — decide as a team how to
  treat those, and write down your choice.

**Hints:**
- Reuse the pipeline you already built: CLAHE → foreground (modal-background
  threshold) → two largest connected components → Otsu inside tissue.
- Bake in the best `kernel_size` and `clip_limit` you found in Exercise 3. If
  different stains preferred different settings, it's fine for the function to
  use different values depending on `stain`.
- Keep the function *pure*: it takes an image (and stain) and returns a mask. It
  shouldn't read files or make plots — that keeps it reusable across jobs 2 and 3.

# imports
import numpy as np
from scipy import ndimage as ndi
from skimage.exposure import equalize_adapthist
from skimage.filters import threshold_otsu


def segment_section(raw, stain, kernel_size=64, clip_limit=0.01):
    raw = raw.astype(np.uint8)
    # converts the image to 8-bit integers => consistent data type

    # Convert RGB/RGBA input into one 2D image.
    if raw.ndim == 3:
        raw = raw[..., :3].mean(axis=2).astype(np.uint8)

    # Accepts grayscale, rgb, or rgba input
    # For RGB/RGBA images

    enhanced = equalize_adapthist(
        raw,
        clip_limit=clip_limit,
        kernel_size=kernel_size
    ) * 255.0
    # Uses CLAHE to create contrast for cortex
    # equalize adapt returns values between 0
    # and 1 then multiplies by 255 to convert to
    # origianl intensity range

    if stain == "myelin":
        work = enhanced.max() - enhanced
    elif stain == "nissl":
        work = enhanced
    elif stain == "nissl+myelin":
        work = enhanced
    else:
        raise ValueError(f"Unknown stain: {stain}")

    # Makes grey matter the Dark class regardless of stain type
    # Nissel: grey matter is allready dark
    # Myelin: grey matter is bright, so invert the image
    # Lets us use one consistent Otsu rule

    values, counts = np.unique(raw, return_counts=True)
    background_value = values[np.argmax(counts)]
    # Estimate background intensity by finding most common pixel value
    # pixels darker than this become tissue

    foreground_threshold = max(0, int(background_value)-10)
    foreground = raw < foreground_threshold
    # Sets a foreground threshold a little below the background value
    # Everything darker than this threshold is considered tissue

    labels, n = ndi.label(foreground)
    # Label all connected tissue regions

    if n == 0:
        return np.zeros_like(raw, dtype=bool)
        # if no tissue is found, retun an empty mask

    sizes = ndi.sum(np.ones_like(labels), labels, range(1, n + 1))
    # meausre size of each connected component

    keep = np.argsort(sizes)[::-1][:2] + 1
    tissue = np.isin(labels, keep)
    # keep two largest components the right and left hemispheres

    if tissue.sum() == 0:
        return np.zeros_like(raw, dtype=bool)
    # if no tissue remains after filtering, return empty mask

    t = threshold_otsu(work[tissue])
    # compute otsu threshold using only pixels inside tissue
    # as to avoid bright slide background that affects threshold

    cortex = tissue & (work < t)
    # grey matter standardizes to darker class and keeps all tissue
    # pixels betlow Otsu threshold

    return cortex

    # return the final binary cortex mask


if __name__ == "__main__":
    import imageio.v3 as iio
    import matplotlib.pyplot as plt

    img = iio.imread("~/sections/yellow_mongoose_nissl/240mngs.jpg")
    mask = segment_section(img, "nissl")

    print(mask.shape)
    print(mask.dtype)

    plt.imshow(mask, cmap="gray")
    plt.show()
print("Raw shape inside function:", raw.shape) 

Job 2 — Validate against the manual data
----------------------------------------

**Task:** for every row that HAS a manual label, run `segment_section`, compare
to the manual label with the Dice score, and summarise how well you're doing.

**Hints:**
- Filter the table to rows with a manual label:

      have_manual = df[df["manual"].notna()]

- For each of those rows: load the raw section, run `segment_section` with that
  row's stain, load the manual label, and compute Dice (reuse your Dice
  function).
- Summarise the results — at least the **mean Dice per species** and **per
  stain**, and the overall mean. This is your evidence for how much to trust the
  automatic method.
- Save the per-section validation scores to a CSV so you can inspect the worst
  cases. Look at a few of them: is the method failing, or is that section just
  hard?

**This is the checkpoint.** Don't move on to Job 3 until the validation numbers
are as good as you can reasonably get them. If the Dice is poor for one stain or
one species, that's a signal to go back and adjust Job 1 (settings, or how you
handle that stain). The whole point of having manual labels is to catch problems
here, before you generate a thousand segmentations you can't check.


Job 3 — Apply to the remaining sections and save
------------------------------------------------

**Task:** for every row that does NOT have a manual label, run `segment_section`
and save the result as a two-layer TIFF in the same format as the `_manual`
files.

**Hints:**
- Filter to the rows still needing segmentation:

      need_seg = df[df["manual"].isna()]

- The output format must match the manual files: a TIFF with **two layers** —
  page 0 = the segmentation, page 1 = the raw image. Write a `write_annotation`
  function so every output is saved the same way:

      import tifffile, numpy as np
      def write_annotation(path, seg_mask, raw):
          seg = (seg_mask > 0).astype(np.uint8) * 255      # 0/255
          tifffile.imwrite(path, np.stack([seg, raw.astype(np.uint8)]))

- Choose a consistent output filename per section. A sensible choice mirrors the
  manual naming — e.g. `<stem>_auto.tif` (use `_auto`, not `_manual`, so it's
  obvious these are automatic, not hand-drawn). Save each next to its section or
  in an output folder you agree on.
- Print progress as you go, and wrap each section in try/except so one bad image
  doesn't halt the whole run.

**Sanity check:** open a few of your output TIFFs (you can reuse
`read_annotation` and display both layers) and confirm the segmentation layer
looks right and the raw layer is intact. Spot-check one Nissl and one myelin.


Putting it together
-------------------

Your finished script should be runnable in one go and do all three jobs in
order: validate first (so the accuracy is reported up front), then apply to the
unlabelled sections. A rough skeleton of `main()`:

    df = pd.read_csv("sect_info_with_manual.csv")

    # Job 2: validate on rows that have a manual label
    validation = validate(df[df["manual"].notna()])
    print_summary(validation)          # mean Dice by species and stain

    # Job 3: apply to rows without a manual label
    apply_to_unlabelled(df[df["manual"].isna()])


Questions to discuss as a team when you're done
-----------------------------------------------

1. What was your overall Dice, and did it differ a lot between Nissl and myelin,
   or between species? What does that tell you about where the method is
   trustworthy and where it isn't?
2. The sections you applied the method to (Job 3) have **no** manual label to
   check against. Given your validation results, how confident are you in those
   outputs? Are there species where you'd want a human to review them before
   using them?
3. If you had one more week, what single change would most improve the weakest
   part of the pipeline? (This is exactly the kind of question the next stage of
   the project — training a neural network — is meant to answer.)
