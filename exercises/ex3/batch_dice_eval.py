#!/usr/bin/env python3
"""
batch_dice_eval.py
==================

Evaluate the automatic cortex segmentation against manual annotations across a
whole directory of files, and summarise Dice scores by species.

What it does:
  1. Searches a directory (recursively) for *_manual.tif annotation files.
  2. Matches each to a row in sect_info.csv by BASE FILENAME (minus extension) --
     the stored paths won't match your local paths, so only the basename is used.
  3. Runs the Otsu cortex segmentation (with CLAHE local-contrast enhancement)
     and computes the Dice score against the manual layer.
  4. Collects everything in a pandas DataFrame.
  5. Prints the average Dice score per species.

Usage:
    python batch_dice_eval.py ANNOTATION_DIR sect_info.csv
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.exposure import equalize_adapthist


# ---------------------------------------------------------------------------
# Reading the two-layer annotation TIFF (top = segmentation, bottom = raw)
# ---------------------------------------------------------------------------
def read_annotation(tif_path):
    with tifffile.TiffFile(tif_path) as tf:
        seg_layer = tf.pages[0].asarray()
        raw_layer = tf.pages[1].asarray()
    return seg_layer, raw_layer


# ---------------------------------------------------------------------------
# The segmentation pipeline (matches the earlier exercise, plus CLAHE).
# `stain` controls grey-matter polarity: in Nissl GM is dark, in myelin GM is
# bright. We flip so GM is always the darker class, then keep below Otsu.
# ---------------------------------------------------------------------------
def segment_cortex(raw, stain):
    raw = raw.astype(np.uint8)

    # CLAHE local contrast enhancement. equalize_adapthist returns a float image
    # in [0, 1]; rescale to 0-255 for consistency with the thresholding below.
    enhanced = equalize_adapthist(raw, clip_limit=0.01) * 255.0

    # Flip so that grey matter is the DARK class regardless of stain type.
    # Nissl: GM already dark -> flip makes GM bright, so DON'T flip for Nissl?
    # We standardise on "GM = dark after this step":
    #   - Nissl  : GM dark  -> keep as is
    #   - myelin : GM bright -> flip so GM becomes dark
    if stain == "myelin":
        work = enhanced.max() - enhanced
    else:  # nissl (and treat nissl+myelin as nissl-like by default)
        work = enhanced

    # Foreground via modal-background threshold (on the ORIGINAL, where the
    # bright paper background is a clean mode).
    values, counts = np.unique(raw, return_counts=True)
    background_value = values[np.argmax(counts)]
    # tissue is darker than background in the original image
    foreground = raw < (background_value - 10)

    # Keep the two largest connected components (the hemispheres).
    labels, n = ndi.label(foreground)
    if n == 0:
        return np.zeros_like(raw, dtype=bool)
    sizes = ndi.sum(np.ones_like(labels), labels, range(1, n + 1))
    keep = np.argsort(sizes)[::-1][:2] + 1
    tissue = np.isin(labels, keep)

    # Otsu inside tissue; GM is the darker class in `work`, so keep BELOW.
    if tissue.sum() == 0:
        return np.zeros_like(raw, dtype=bool)
    t = threshold_otsu(work[tissue])
    cortex = tissue & (work < t)
    return cortex


def dice_score(pred, truth):
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    denom = pred.sum() + truth.sum()
    if denom == 0:
        return np.nan
    return 2.0 * np.logical_and(pred, truth).sum() / denom


# ---------------------------------------------------------------------------
# Build a lookup from base filename (minus extension) -> list of CSV rows.
# A stem can map to MORE THAN ONE row (the same basename exists in both the
# nissl and myelin folders for some specimens), so we keep a list and try to
# disambiguate by a stain hint in the annotation's folder path.
# ---------------------------------------------------------------------------
def build_lookup(csv_path):
    df = pd.read_csv(csv_path)
    df["stem"] = df["filename"].apply(
        lambda f: os.path.splitext(os.path.basename(str(f)))[0]
    )
    lookup = {}
    for _, row in df.iterrows():
        lookup.setdefault(row["stem"], []).append(row)
    return df, lookup


def pick_row(candidates, annotation_path):
    """Given >=1 candidate CSV rows for a stem, choose the best match.
    If there's only one, use it. If several differ by stain, use a stain hint
    from the annotation's folder path (e.g. '.../cat_myelin/..._manual.tif')."""
    if len(candidates) == 1:
        return candidates[0], None
    path_l = annotation_path.lower()
    for stain in ("myelin", "nissl"):
        if stain in path_l:
            for c in candidates:
                if c["stain"] == stain:
                    return c, None
    # couldn't disambiguate -> take the first but flag it
    return candidates[0], (
        f"ambiguous stem maps to {len(candidates)} rows "
        f"(stains: {sorted(set(c['stain'] for c in candidates))}); "
        f"no stain hint in path, used first"
    )


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1
    annotation_dir, csv_path = argv[0], argv[1]

    df, lookup = build_lookup(csv_path)

    # 1) find every *_manual.tif under the directory (recursively)
    pattern = os.path.join(annotation_dir, "**", "*_manual.tif")
    files = sorted(glob.glob(pattern, recursive=True))
    print(f"Found {len(files)} *_manual.tif file(s) under {annotation_dir}\n")

    records = []
    for path in files:
        stem = os.path.basename(path)[:-len("_manual.tif")]
        candidates = lookup.get(stem)
        if not candidates:
            print(f"  no CSV match for stem {stem!r} ({os.path.basename(path)})")
            records.append(dict(stem=stem, species=None, stain=None,
                                dice=np.nan, note="no CSV match"))
            continue

        row, note = pick_row(candidates, path)
        try:
            manual_layer, raw = read_annotation(path)
            manual_mask = manual_layer > 0
            cortex = segment_cortex(raw, row["stain"])
            d = dice_score(cortex, manual_mask)
        except Exception as e:
            records.append(dict(stem=stem, species=row["species"],
                                stain=row["stain"], dice=np.nan,
                                note=f"error: {e}"))
            print(f"  ERROR on {stem}: {e}")
            continue

        records.append(dict(stem=stem, species=row["species"],
                            stain=row["stain"], dice=d, note=note))
        flag = f"  [{note}]" if note else ""
        print(f"  {stem:24s} {row['species']:24s} {row['stain']:6s} "
              f"Dice={d:.3f}{flag}")

    results = pd.DataFrame.from_records(records)

    # 4) average Dice by species
    print("\n" + "=" * 60)
    print("Average Dice by species")
    print("=" * 60)
    if results["dice"].notna().any():
        by_species = (results.dropna(subset=["dice"])
                      .groupby("species")["dice"]
                      .agg(["mean", "count"])
                      .sort_values("mean", ascending=False))
        print(by_species.to_string(float_format=lambda x: f"{x:.3f}"))
        print(f"\nOverall mean Dice: {results['dice'].mean():.3f} "
              f"(n={results['dice'].notna().sum()})")
    else:
        print("No valid Dice scores computed.")

    # save the full table
    out_csv = "dice_results.csv"
    results.to_csv(out_csv, index=False)
    print(f"\nSaved per-file results to {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
