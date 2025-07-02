import os, rasterio, numpy as np

RAW_DIR = "data/"       
PATCH_DIR = "data/patches"   
SIZE = 64
STRIDE = 64

months = ['jan','feb']
# months = ['jan','feb','mar','apr','may','jun',
#           'jul','aug','sep','oct','nov','dec']

os.makedirs(PATCH_DIR, exist_ok=True)

for m in months:
    tif = f"{RAW_DIR}/ndvi_{m}_2023.tif"
    out = f"{PATCH_DIR}/{m}"
    os.makedirs(out, exist_ok=True)

    with rasterio.open(tif) as src:
        img = src.read(1).astype(np.float32)
        img[img < -0.2] = np.nan

    H, W = img.shape; pid = 0
    for y in range(0, H-SIZE+1, STRIDE):
        for x in range(0, W-SIZE+1, STRIDE):
            patch = img[y:y+SIZE, x:x+SIZE]
            if np.isnan(patch).all():
                continue
            np.save(f"{out}/{pid:05d}.npy", patch)
            pid += 1
    print(f"{m}: saved {pid} patches")
