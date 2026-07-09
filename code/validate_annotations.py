#!/usr/bin/env python3
"""
validate_annotations.py
=======================

Validate the two-layer annotation TIFFs produced by the interns.

Each annotation is a multi-page TIFF with exactly two layers:
  - TOP layer  (page 0): the segmentation -- should contain only TWO pixel
                         values (foreground and background).
  - BOTTOM layer (page 1): the raw section image.

This script checks that each file is well-formed and reports any problems.

Usage
-----
    # check one file
    python validate_annotations.py path/to/annotation.tif

    # check several files or a whole folder of .tif files
    python validate_annotations.py folder/*.tif
    python validate_annotations.py file1.tif file2.tif file3.tif

Exit code is 0 if every file passed, 1 if any file failed -- so this can be
used in scripts or continuous-integration checks.
"""

import sys
import glob
import numpy as np
import tifffile


def read_annotation(tif_path):
    """Open a two-layer annotation TIFF and return its two layers as separate
    NumPy arrays.

    Parameters
    ----------
    tif_path : str
        Path to the annotation TIFF.

    Returns
    -------
    seg_layer : np.ndarray
        The segmentation layer (top layer / page 0).
    raw_layer : np.ndarray
        The raw section image (bottom layer / page 1).

    Raises
    ------
    ValueError
        If the file does not have exactly two layers.
    """
    with tifffile.TiffFile(tif_path) as tf:
        pages = tf.pages
        if len(pages) != 2:
            raise ValueError(
                f"expected exactly 2 layers, found {len(pages)}"
            )
        seg_layer = pages[0].asarray()
        raw_layer = pages[1].asarray()
    return seg_layer, raw_layer


def validate_one(tif_path):
    """Validate a single annotation TIFF. Returns a list of problem strings
    (empty list means the file passed)."""
    problems = []

    # 1) can we open it and does it have two layers?
    try:
        seg, raw = read_annotation(tif_path)
    except FileNotFoundError:
        return ["file not found"]
    except ValueError as e:
        return [str(e)]
    except Exception as e:
        return [f"could not read file: {e}"]

    # 2) the two layers should be the same height and width
    if seg.shape[:2] != raw.shape[:2]:
        problems.append(
            f"layer sizes differ: segmentation {seg.shape[:2]} "
            f"vs raw {raw.shape[:2]}"
        )

    # 3) the segmentation layer should contain only TWO distinct values
    #    (foreground and background). If a layer might be colour/multi-channel
    #    we compare the top layer's unique values.
    seg_values = np.unique(seg)
    if len(seg_values) != 2:
        problems.append(
            f"segmentation layer should have exactly 2 values "
            f"(foreground/background) but has {len(seg_values)}: "
            f"{seg_values[:8]}{'...' if len(seg_values) > 8 else ''}"
        )

    # 4) sanity check that we didn't swap the layers: the segmentation (top)
    #    should have FEWER distinct values than the raw image (bottom). If the
    #    top layer looks like a photo and the bottom like a mask, they're
    #    probably in the wrong order.
    if len(seg_values) > len(np.unique(raw)):
        problems.append(
            "top layer has more distinct values than the bottom layer -- "
            "the segmentation and raw layers may be in the wrong order"
        )

    return problems


def main(argv):
    # expand any glob patterns (e.g. "folder/*.tif") and collect file paths
    paths = []
    for arg in argv:
        matched = glob.glob(arg)
        paths.extend(matched if matched else [arg])

    if not paths:
        print("No files given.\n")
        print(__doc__)
        return 1

    all_ok = True
    for path in paths:
        problems = validate_one(path)
        if problems:
            all_ok = False
            print(f"FAIL  {path}")
            for p in problems:
                print(f"        - {p}")
        else:
            print(f"OK    {path}")

    print()
    if all_ok:
        print(f"All {len(paths)} file(s) passed.")
        return 0
    else:
        print("Some files failed validation (see above).")
        return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
