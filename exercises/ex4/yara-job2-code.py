from pathlib import Path
import numpy as np
import pandas as pd
import tifffile


def dice_score(mask_a, mask_b):
    """Dice similarity for two binary masks."""
    a = np.asarray(mask_a, dtype=bool)
    b = np.asarray(mask_b, dtype=bool)

    total = a.sum() + b.sum()
    if total == 0:
        return 1.0

    return 2 * np.logical_and(a, b).sum() / total


def read_annotation(tif_path):
    """Return (segmentation_mask, raw_image) from a two-layer TIFF."""
    layers = tifffile.imread(tif_path)

    if layers.ndim < 3 or layers.shape[0] < 2:
        raise ValueError(f"Expected a two-layer TIFF: {tif_path}")

    seg_layer = layers[0] > 0
    raw_layer = layers[1]
    return seg_layer, raw_layer


def validate(manual_rows, data_dir="."):
    """Segment every manually labelled section and return Dice results."""
    results = []

    for _, row in manual_rows.iterrows():
        try:
            raw_path = Path(data_dir) / row["filename"]
            manual_path = Path(data_dir) / row["manual"]

            raw = tifffile.imread(raw_path)
            manual_mask, _ = read_annotation(manual_path)

            auto_mask = segment_section(raw, row["stain"])
            score = dice_score(auto_mask, manual_mask)

            results.append({
                "specimen": row["specimen"],
                "species": row["species"],
                "stain": row["stain"],
                "section": row["section"],
                "filename": row["filename"],
                "manual": row["manual"],
                "dice": score,
            })
            print(f"Validated {row['filename']}: Dice = {score:.3f}")

        except Exception as error:
            print(f"Could not validate {row['filename']}: {error}")

    return pd.DataFrame(results)


def print_validation_summary(validation):
    print(f"\nOverall mean Dice: {validation['dice'].mean():.3f}")

    print("\nMean Dice by species:")
    print(validation.groupby("species")["dice"].mean().sort_values())

    print("\nMean Dice by stain:")
    print(validation.groupby("stain")["dice"].mean().sort_values())


df = pd.read_csv("sect_info_with_manual.csv")

have_manual = df[df["manual"].notna()]
validation = validate(have_manual, data_dir=".")

validation.to_csv("validation_scores.csv", index=False)
print_validation_summary(validation)

print("\nWorst sections:")
print(validation.nsmallest(10, "dice")[["filename", "species", "stain", "dice"]])

print(validation.nsmallest(10, "dice")[["filename", "species", "stain", "dice"]])%  
