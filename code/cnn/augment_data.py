#!/usr/bin/env python3
"""
augment_data.py
===============

Data augmentation for the image/label pairs used to train the segmentation
network. For every image in images/ and its matching label in label/, this
creates several randomly transformed copies, expanding a small dataset into a
larger, more varied one.

    OUTPUT/images/<name>_augN_0000.nii.gz
    OUTPUT/label/<name>_augN.nii.gz

### THE KEY RULE OF SEGMENTATION AUGMENTATION ###
There are two kinds of transform, and they are treated differently:

  * GEOMETRIC (rotate, flip, zoom, shift): change WHERE things are. These MUST
    be applied to the image AND the label in EXACTLY the same way, or the label
    will no longer line up with the image. The label is interpolated with
    NEAREST-NEIGHBOUR so it stays a clean 0/1 mask (never blurred).

  * INTENSITY (noise, brightness/contrast): change what the image LOOKS like,
    not where things are. These are applied to the IMAGE ONLY. Adding noise to a
    label would be meaningless and would corrupt it.

Getting this distinction wrong is the classic augmentation bug: either the label
drifts out of alignment with the image, or the mask gets blurred into non-binary
values. This script keeps them straight.

Usage:
    python augment_data.py INPUT_DIR OUTPUT_DIR [--n 5] [--seed 0]

INPUT_DIR must contain images/ and label/ subfolders (as produced by
extract_annotations.py). --n is how many augmented copies to make per section.
"""

import os
import glob
import argparse
import numpy as np
import nibabel as nib
from scipy import ndimage as ndi


# ---------------------------------------------------------------------------
# Geometric transforms: applied to BOTH image and label together.
# The image uses smooth (linear, order=1) interpolation; the label uses
# nearest-neighbour (order=0) so it stays binary.
# ---------------------------------------------------------------------------
def random_geometric(image, label, rng):
    """Apply the SAME random rotation, flip, zoom, and shift to image + label."""
    img, lab = image, label

    # --- random horizontal flip ---
    if rng.random() < 0.5:
        img = img[:, ::-1]
        lab = lab[:, ::-1]

    # --- random rotation (small angle, so the brain stays roughly upright) ---
    angle = rng.uniform(-20, 20)                       # degrees
    img = ndi.rotate(img, angle, reshape=False, order=1, mode="nearest")
    lab = ndi.rotate(lab, angle, reshape=False, order=0, mode="nearest")

    # --- random zoom (scale in/out, then crop/pad back to the original size) ---
    zoom = rng.uniform(0.85, 1.15)
    img = _zoom_keep_size(img, zoom, order=1)
    lab = _zoom_keep_size(lab, zoom, order=0)

    # --- random shift (translate a few percent of the image) ---
    max_shift = 0.06 * np.array(img.shape)
    dy, dx = rng.uniform(-max_shift, max_shift)
    img = ndi.shift(img, (dy, dx), order=1, mode="nearest")
    lab = ndi.shift(lab, (dy, dx), order=0, mode="nearest")

    return img, lab


def _zoom_keep_size(arr, factor, order):
    """Zoom an array by `factor` but return it at the ORIGINAL size, by
    center-cropping (if zoomed in) or center-padding (if zoomed out)."""
    h, w = arr.shape
    zoomed = ndi.zoom(arr, factor, order=order, mode="nearest")
    zh, zw = zoomed.shape
    out = np.zeros_like(arr)
    if factor >= 1.0:
        # zoomed in -> crop the center back to (h, w)
        top = (zh - h) // 2
        left = (zw - w) // 2
        out = zoomed[top:top + h, left:left + w]
    else:
        # zoomed out -> paste into the center of a blank canvas
        top = (h - zh) // 2
        left = (w - zw) // 2
        out[top:top + zh, left:left + zw] = zoomed
    return out


# ---------------------------------------------------------------------------
# Intensity transforms: applied to the IMAGE ONLY. The label is not touched.
# ---------------------------------------------------------------------------
def random_intensity(image, rng):
    """Apply random brightness/contrast and noise to the image only."""
    img = image.astype(np.float32)

    # --- brightness and contrast ---
    contrast = rng.uniform(0.85, 1.15)
    brightness = rng.uniform(-0.08, 0.08)
    mean = img.mean()
    img = (img - mean) * contrast + mean + brightness

    # --- additive Gaussian noise ---
    if rng.random() < 0.7:
        sigma = rng.uniform(0.01, 0.05)
        img = img + rng.normal(0, sigma, img.shape)

    # keep values in the valid [0, 1] range
    return np.clip(img, 0.0, 1.0)


def augment_pair(image, label, rng):
    """One augmented copy: geometric transform on both, then intensity on the
    image alone."""
    img, lab = random_geometric(image, label, rng)
    img = random_intensity(img, rng)
    # the label may have collected tiny interpolation artefacts at edges from
    # the flip's negative-stride view; make sure it's a clean 0/1 array
    lab = (np.asarray(lab) > 0.5).astype(np.uint8)
    return img.astype(np.float32), lab


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_dir", help="folder containing images/ and label/")
    ap.add_argument("output_dir", help="where to write augmented images/ and label/")
    ap.add_argument("--n", type=int, default=5,
                    help="augmented copies per section (default 5)")
    ap.add_argument("--seed", type=int, default=0, help="random seed")
    ap.add_argument("--keep-originals", action="store_true",
                    help="also copy the un-augmented originals into the output")
    args = ap.parse_args(argv)

    images_in = os.path.join(args.input_dir, "images")
    labels_in = os.path.join(args.input_dir, "label")
    images_out = os.path.join(args.output_dir, "images")
    labels_out = os.path.join(args.output_dir, "label")
    os.makedirs(images_out, exist_ok=True)
    os.makedirs(labels_out, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    image_paths = sorted(glob.glob(os.path.join(images_in, "*_0000.nii.gz")))
    print(f"Found {len(image_paths)} section(s); making {args.n} "
          f"augmented copies each\n")

    n_written = 0
    for ip in image_paths:
        name = os.path.basename(ip).replace("_0000.nii.gz", "")
        lp = os.path.join(labels_in, name + ".nii.gz")
        if not os.path.exists(lp):
            print(f"  skip {name}: no matching label")
            continue

        image = np.squeeze(nib.load(ip).get_fdata()).astype(np.float32)
        label = np.squeeze(nib.load(lp).get_fdata())

        if args.keep_originals:
            nib.save(nib.Nifti1Image(image, np.eye(4)),
                     os.path.join(images_out, f"{name}_0000.nii.gz"))
            nib.save(nib.Nifti1Image((label > 0.5).astype(np.uint8), np.eye(4)),
                     os.path.join(labels_out, f"{name}.nii.gz"))
            n_written += 1

        for k in range(args.n):
            aug_img, aug_lab = augment_pair(image, label, rng)
            nib.save(nib.Nifti1Image(aug_img, np.eye(4)),
                     os.path.join(images_out, f"{name}_aug{k}_0000.nii.gz"))
            nib.save(nib.Nifti1Image(aug_lab, np.eye(4)),
                     os.path.join(labels_out, f"{name}_aug{k}.nii.gz"))
            n_written += 1
        print(f"  {name}: wrote {args.n} augmented copies")

    print(f"\nWrote {n_written} image/label pair(s) to {args.output_dir}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
