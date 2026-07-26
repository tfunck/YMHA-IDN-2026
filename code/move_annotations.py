from pathlib import Path
import shutil
import pandas as pd

df = pd.read_csv('../sect_info_with_manual.csv')
tmpsegment = df[df["manual"].isna()]
auto_segments = tmpsegment[['filename']]
# print (auto_segments)
for auto_seg in auto_segments.itertuples():
    fname = '../../'+auto_seg.filename
    print("Moving:", fname)
    infilename = fname.replace('.jpg','.tif')
    outfilename = '../../v.1/annotations/' + Path(infilename).name
    shutil.move(infilename, outfilename)
