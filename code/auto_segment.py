import pandas as pd
import tifffile
import numpy as np
from imageio.v3 import imread
from segment_section import segment_section
from skimage.color import rgb2gray


def write_annotation(path, seg_mask, raw):
    seg = (seg_mask > 0).astype(np.uint8) * 255

    if raw.ndim == 3: 
    #if the image has 3 color channels, it is converted to grayscale
        raw = (rgb2gray(raw) * 255).astype(np.uint8)
    #convert rgb images to 8 bit grayscale so thye match the annotation form
    else:
        raw = raw.astype(np.uint8)
    # if the image is already in grayscale, it is converted to an 8bit format
    tifffile.imwrite(path, np.stack([seg, raw.astype(np.uint8)]))
    # save a two later tiff with the segmentation mask and original grayscale


def auto_section(csvfile='sect_info_with_manual.csv'):
    df = pd.read_csv(csvfile)
    tmpsegment = df[df["manual"].isna()]
    auto_segments = tmpsegment[['filename', 'stain']]
    # print (auto_segments)
    for auto_seg in auto_segments.itertuples():
        print("Processing:", auto_seg.filename)
        raw = imread(auto_seg.filename) 
        #read the original image so segment_section can process it
        mask = segment_section(raw, auto_seg.stain)
        #generate the cortex mask form the give image
        outpath = auto_seg.filename.replace('.jpg', '.tif')
        write_annotation(outpath, mask, raw)
        #save segmentation mask and original image into the TIFF annotation
