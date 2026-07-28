"""
make_demo_data.py -- generate a small synthetic dataset to practise on.

Creates the same folder layout the real data uses:
    demo_data/images/<name>_0000.nii.gz   (raw sections)
    demo_data/label/<name>.nii.gz         (grey-matter labels)

Each "section" is a simple synthetic brain: a bright cortical ribbon (the grey
matter we want to segment) around a white-matter core, on a dark background.
This lets you run cnn_segmentation.py without the real data.
"""
import os
import numpy as np
import nibabel as nib

os.makedirs("demo_data/images", exist_ok=True)
os.makedirs("demo_data/label", exist_ok=True)

def make_section(seed):
    r = np.random.default_rng(seed)
    H = W = 96
    img = np.full((H, W), 0.15)                     # dark background
    yy, xx = np.ogrid[:H, :W]
    cx, cy = W // 2 + r.integers(-6, 6), H // 2 + r.integers(-6, 6)
    rad = r.integers(30, 38)
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img[dist < rad] = 0.5                            # white matter
    ribbon = (dist < rad) & (dist > rad - 7)
    img[ribbon] = 0.8                               # cortex (grey matter)
    img = np.clip(img + r.normal(0, 0.05, (H, W)), 0, 1).astype(np.float32)
    label = ribbon.astype(np.uint8)
    return img, label

for i in range(12):
    img, lab = make_section(i)
    nib.save(nib.Nifti1Image(img, np.eye(4)), f"demo_data/images/sec{i:02d}_0000.nii.gz")
    nib.save(nib.Nifti1Image(lab, np.eye(4)), f"demo_data/label/sec{i:02d}.nii.gz")

print("Wrote 12 image/label pairs into demo_data/")
