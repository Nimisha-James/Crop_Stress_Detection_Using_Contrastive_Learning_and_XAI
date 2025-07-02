# 1. VIEW EACH PATCH

# import numpy as np
# import matplotlib.pyplot as plt

# # Load patch (e.g. from jan folder)
# patch = np.load("data/patches/jan/00012.npy")  # shape (64, 64)

# # Optional: mask very low NDVI (e.g. clouds or no data)
# patch = np.where(patch < -0.2, np.nan, patch)

# # Plot with color scale
# plt.imshow(patch, cmap='RdYlGn', vmin=0, vmax=1)
# plt.colorbar(label="NDVI")
# plt.title("NDVI Patch - Jan")
# plt.axis('off')
# plt.show()


# 2. VIEW PATCH SERIES
import numpy as np, matplotlib.pyplot as plt
cube = np.load('../data/series/00000.npy')   # [2,64,64]
# cube = (cube + 1) / 2                         # -1‒1 ➜ 0‒1

fig, axes = plt.subplots(1, 2, figsize=(6,3))
for i in range(2):
    axes[i].imshow(cube[i], cmap='YlGn', vmin=0, vmax=1)
    axes[i].set_title(f"Month {i+1}")
    axes[i].axis('off')
plt.tight_layout(); plt.show()
