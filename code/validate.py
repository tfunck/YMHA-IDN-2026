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
