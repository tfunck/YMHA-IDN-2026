#!/usr/bin/env python3
"""
build_manual_labels.py
======================

Walk a sections directory for the interns' *_manual.tif annotation files,
extract and binarize each segmentation layer, save it as a companion
*_manual_label.tif, and record that label file in a new "manual" column of
sect_info.csv (matched to each section by its base filename).

Each annotation is a two-layer TIFF:
  - page 0 = segmentation layer
  - page 1 = raw image

Usage:
    python build_manual_labels.py SECTIONS_DIR sect_info.csv [-o OUTPUT_CSV]

If -o is omitted, the updated table is written to sect_info_with_manual.csv
(the original CSV is left untouched).
"""

import sys
import os
import glob
import argparse
import numpy as np
import pandas as pd
import tifffile


def stem_of(path):
    """Base filename with no directory and no extension."""
    return os.path.splitext(os.path.basename(str(path)))[0]


def parent_of(path):
    """Name of the immediate parent directory (e.g. 'cat_myelin')."""
    return os.path.basename(os.path.dirname(str(path)))


def read_segmentation_layer(tif_path):
    """Return the segmentation layer (page 0) of a two-layer annotation TIFF."""
    with tifffile.TiffFile(tif_path) as tf:
        return tf.pages[0].asarray()


def binarize(seg):
    """Turn a segmentation layer into a clean 0/255 uint8 mask.
    Any non-zero pixel becomes foreground (255); zero stays background (0).
    This is robust whether the input was already 0/1, 0/255, or something else."""
    return (np.asarray(seg) > 0).astype(np.uint8) * 255


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sections_dir", help="directory to search for *_manual.tif")
    ap.add_argument("csv_path", help="path to sect_info.csv")
    ap.add_argument("-o", "--output", default="sect_info_with_manual.csv",
                    help="where to write the updated CSV "
                         "(default: sect_info_with_manual.csv)")
    args = ap.parse_args(argv)

    # --- load the section table and build lookups ---
    df = pd.read_csv(args.csv_path)
    df["manual"] = pd.NA          # new column, empty to start

    # Some basenames appear in more than one row (the same section exists under
    # both a *_nissl and a *_myelin folder). To tell those apart we key on
    # (parent_directory, stem) as well as on stem alone. The parent directory in
    # the CSV's filename column (e.g. 'cat_myelin') encodes the stain, and the
    # intern's annotation lives in a matching folder, so we can disambiguate.
    stem_to_rows = {}                 # stem -> [row indices]
    parent_stem_to_rows = {}          # (parent, stem) -> [row indices]
    for idx, fname in df["filename"].items():
        s = stem_of(fname)
        p = parent_of(fname)
        stem_to_rows.setdefault(s, []).append(idx)
        parent_stem_to_rows.setdefault((p, s), []).append(idx)

    # --- find every *_manual.tif under the sections directory (recursively) ---
    pattern = os.path.join(args.sections_dir, "**", "*_manual.tif*")
    manual_files = sorted(glob.glob(pattern, recursive=True))
    print(f"Found {len(manual_files)} *_manual.tif file(s) under "
          f"{args.sections_dir}\n")

    n_written = 0
    n_matched = 0
    n_unmatched = 0
    for path in manual_files:
        # the CSV stem is the annotation stem minus the trailing "_manual"
        ann_stem = stem_of(path)                     # e.g. redroo_sec0100_manual
        csv_stem = ann_stem[:-len("_manual")] if ann_stem.endswith("_manual") \
            else ann_stem                            # e.g. redroo_sec0100

        # --- extract + binarize the segmentation layer, save as _manual_label ---
        try:
            seg = read_segmentation_layer(path)
            label = binarize(seg)
        except Exception as e:
            print(f"  ERROR reading {path}: {e}")
            continue

        label_path = os.path.join(os.path.dirname(path),
                                  f"{csv_stem}_manual_label.tif")
        tifffile.imwrite(label_path, label)
        n_written += 1

        # --- record it in the matching CSV row(s) ---
        rows = stem_to_rows.get(csv_stem)
        if not rows:
            n_unmatched += 1
            print(f"  saved {os.path.basename(label_path)}  "
                  f"(no CSV row matches stem {csv_stem!r})")
            continue

        note = ""
        if len(rows) > 1:
            # ambiguous stem: disambiguate by the annotation's parent directory,
            # which encodes the stain (e.g. 'cat_myelin' vs 'cat_nissl').
            ann_parent = parent_of(path)
            disambig = parent_stem_to_rows.get((ann_parent, csv_stem))
            if disambig:
                rows = disambig
                note = f"  [disambiguated by folder '{ann_parent}']"
            else:
                note = (f"  [AMBIGUOUS: stem in {len(rows)} rows and parent "
                        f"'{ann_parent}' matched none; filled all]")

        for idx in rows:
            df.at[idx, "manual"] = label_path
        n_matched += 1
        target = rows[0] if len(rows) == 1 else rows
        print(f"  saved {os.path.basename(label_path)} -> row {target}{note}")

    # --- write the updated table ---
    df.to_csv(args.output, index=False)
    print(f"\nWrote {n_written} label file(s).")
    print(f"Matched {n_matched} annotation(s) to CSV rows; "
          f"{n_unmatched} had no match.")
    print(f"Rows with a manual label: {df['manual'].notna().sum()} of {len(df)}")
    print(f"Updated table saved to: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
