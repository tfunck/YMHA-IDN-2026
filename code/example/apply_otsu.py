from code.yara import validate_segmentation
from code.bella import segment
from code.ava import job3

import pandas as pd

df = pd.read_csv('sect_info.csv')

for i, row in df.iterrows() :
    path = row['path']

    seg_path = segment(path)
    
    validate_segmentation(seg_path)

    # job3
    job3()


