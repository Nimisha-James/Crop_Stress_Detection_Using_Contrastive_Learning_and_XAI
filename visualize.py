import numpy as np
import matplotlib.pyplot as plt

def show_patch(id):
    cube = np.load(f"data/series/{id:05d}.npy")
    cube = (cube + 1) / 2

    T = cube.shape[0]  # number of months
    rows = (T + 3) // 4  # 4 plots per row

    fig, axes = plt.subplots(rows, 4, figsize=(12, 3 * rows))
    axes = axes.flatten()

    for i in range(T):
        axes[i].imshow(cube[i], cmap="YlGn", vmin=0, vmax=1)
        axes[i].set_title(f"Month {i+1}")
        axes[i].axis('off')

    # Hide unused subplots
    for j in range(T, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()


show_patch(12)
show_patch(57)
show_patch(34)