"""
Utility functions for 3D Piscis.
"""

import numpy as np
from typing import Sequence


def remove_duplicate_coords_3d(
    coords: np.ndarray,
    threshold: float = 1.0
) -> np.ndarray:
    """
    Remove duplicate 3D coordinates within a distance threshold.

    Parameters
    ----------
    coords : np.ndarray
        Coordinates with shape (n_spots, 3) where columns are (z, y, x).
    threshold : float, optional
        Distance threshold. Default is 1.0.

    Returns
    -------
    new_coords : np.ndarray
        Coordinates without duplicates.
    """
    if len(coords) == 0:
        return coords
    
    # Use a simple distance-based approach
    from scipy.spatial.distance import cdist
    
    # Compute pairwise distances
    distances = cdist(coords, coords)
    
    # Find coordinates that are too close (within threshold)
    # Use a greedy approach: keep first coordinate, remove duplicates
    keep = np.ones(len(coords), dtype=bool)
    
    for i in range(len(coords)):
        if not keep[i]:
            continue
        # Mark all coordinates within threshold as duplicates (except itself)
        duplicates = (distances[i] < threshold) & (np.arange(len(coords)) != i)
        keep[duplicates] = False
    
    new_coords = coords[keep]
    
    return new_coords


def pad_3d(images: Sequence[np.ndarray]) -> Sequence[np.ndarray]:
    """
    Pad 3D images to the same size.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        List of 3D images with shape (z, y, x).

    Returns
    -------
    padded_images : Sequence[np.ndarray]
        List of padded images.
    """
    if len(images) == 0:
        return images
    
    # Compute the padded image size for each dimension
    padded_size = [max([image.shape[i] for image in images]) for i in range(3)]
    
    # Pad images
    padded_images = []
    for image in images:
        if image.shape == tuple(padded_size):
            padded_image = image
        else:
            pad_width = [(0, m - n) for m, n in zip(padded_size, image.shape)]
            padded_image = np.pad(image, pad_width, mode='constant', constant_values=0)
        padded_images.append(padded_image)
    
    return padded_images
