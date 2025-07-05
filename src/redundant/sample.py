import os, numpy as np
from pathlib import Path
import rasterio, glob
from collections import defaultdict

DIR = Path("../indices//jan_1_2023.tif")
MONTHS     = ['jan','feb','mar','apr','may','jun', 'jul', 'aug', 'sept']
T=6
PATCH_DIR = "../data/patches"
DEST = "../data/series"

patch_dict = defaultdict(list)

arr = np.load("../data/patches/jan/00001.npy")
print(arr.shape)
# print(tifs)
# print(os.path.basename(tifs[1]).split('_')[1])