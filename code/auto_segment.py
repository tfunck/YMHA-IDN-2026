import pandas as pd
import tifffile, numpy as np
from segment_section.py import segment_section

def write_annotation(path, seg_mask, raw):
    seg = (seg_mask > 0).astype(np.uint8) * 255      # 0/255
    tifffile.imwrite(path, np.stack([seg, raw.astype(np.uint8)]))
    
def auto_section(csvfile = 'sect_info_with_manual.csv'):
    df = pd.read_csv(csvfile)
    tmpsegment = df[df["manual"].isna()]
    auto_segments = tmpsegment [['filename','stain']]
    #print (auto_segments)
    for auto_seg in auto_segments.itertuples():
        mask = segment_section(auto_seg.filename, auto_seg.stain)
        outpath = auto_seg.filename.replace('.jpg','.tif')
        write_annotation(outpath,mask,auto_seg.filename)
