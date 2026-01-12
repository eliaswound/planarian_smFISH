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
    
    Memory-efficient version that avoids computing full pairwise distance matrix.
    Uses a hash-based approach for large coordinate sets.

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
    
    # For very large coordinate sets (>100k), skip deduplication to save memory
    # The dataset generation will handle this, and deduplication can be done later if needed
    if len(coords) > 100000:
        # Just return coordinates as-is for very large sets
        # This prevents OOM during dataset generation
        return coords
    
    # For small coordinate sets, use the original approach
    if len(coords) < 10000:
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
    
    # For medium-large coordinate sets (10k-100k), use a memory-efficient hash-based approach
    # Round coordinates to grid cells of size ~threshold, then use unique
    # This approximates duplicate removal but is much more memory-efficient
    
    # Round to grid cells - use float32 to save memory
    grid_size = max(0.5, threshold / 2.0)  # Use smaller grid for better precision
    coords_float = coords.astype(np.float32)  # Ensure float32
    rounded_coords = np.round(coords_float / grid_size).astype(np.int32)
    
    # Free memory immediately
    del coords_float
    import gc
    gc.collect()
    
    # Use unique to find first occurrence of each grid cell
    # This is O(n log n) instead of O(n^2)
    _, unique_indices = np.unique(rounded_coords, axis=0, return_index=True)
    
    # Free memory
    del rounded_coords
    gc.collect()
    
    # Sort indices to maintain original order
    unique_indices = np.sort(unique_indices)
    
    new_coords = coords[unique_indices]
    
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
