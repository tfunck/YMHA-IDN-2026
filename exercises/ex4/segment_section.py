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
