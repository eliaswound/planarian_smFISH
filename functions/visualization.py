# functions/visualization.py
import numpy as np
import matplotlib.pyplot as plt


def plot_spot_example(img, coord, gaussian_fit=True, save_path="spot_example.png"):
    """
    Visualize a single 3D spot.
    Args:
        img (np.ndarray): 3D image (Z,Y,X)
        coord (tuple/list/np.ndarray): (z, y, x) voxel coordinate
        gaussian_fit (bool): True if the spot passed Gaussian fitting
        save_path (str): Where to save the figure
    """
    z, y, x = coord
    z1, z2 = max(0, z - 2), min(img.shape[0], z + 3)
    y1, y2 = max(0, y - 5), min(img.shape[1], y + 6)
    x1, x2 = max(0, x - 5), min(img.shape[2], x + 6)

    sub = img[z1:z2, y1:y2, x1:x2]

    # Show maximum intensity projection in Z
    mip = sub.max(axis=0)

    plt.figure(figsize=(5, 5))
    plt.imshow(mip, cmap='hot')
    title = f"{'Gaussian' if gaussian_fit else 'Non-Gaussian'} spot at {coord}"
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, dpi=150)
    plt.close()