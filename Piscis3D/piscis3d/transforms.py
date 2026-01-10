"""
3D Transformations for Piscis.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy import ndimage
from typing import Any, List, Optional, Sequence, Tuple
from jax import jit, vmap


def batch_adjust_3d(
    images: Sequence[np.ndarray],
    adjustment: Optional[str],
    **kwargs: Any
) -> Sequence[np.ndarray]:
    """
    Batch adjust 3D images.

    Parameters
    ----------
    images : Sequence[np.ndarray]
        Images to adjust with shape (z, y, x).
    adjustment : Optional[str]
        Adjustment type. Supported types are 'normalize' and 'standardize'.
    **kwargs : Any
        Keyword arguments for the adjustment function.

    Returns
    -------
    adjusted_images : Sequence[np.ndarray]
        Adjusted images.
    """
    if adjustment is not None:
        images = list(images)
        adjusted_images = np.empty(len(images), dtype=object)
        for i, image in enumerate(images):
            adjusted_images[i] = adjust_3d(image, adjustment, **kwargs)
    else:
        adjusted_images = images
    
    return adjusted_images


def adjust_3d(
    image: np.ndarray,
    adjustment: Optional[str],
    **kwargs: Any
) -> np.ndarray:
    """
    Adjust a 3D image.

    Parameters
    ----------
    image : np.ndarray
        3D image with shape (z, y, x).
    adjustment : Optional[str]
        Adjustment type. Supported types are 'normalize' and 'standardize'.
    **kwargs : Any
        Keyword arguments for the adjustment function.

    Returns
    -------
    adjusted_image : np.ndarray
        Adjusted image.
    """
    if adjustment is None:
        adjusted_image = image
    elif adjustment == 'normalize':
        adjusted_image = normalize_3d(image, **kwargs)
    elif adjustment == 'standardize':
        adjusted_image = standardize_3d(image, **kwargs)
    else:
        raise ValueError(f"Adjustment type '{adjustment}' is not supported.")
    
    return adjusted_image


def normalize_3d(
    image: np.ndarray,
    lower: float = 0,
    upper: float = 100,
    epsilon: float = 1e-7
) -> np.ndarray:
    """
    Normalize a 3D image to the range [0, 1] based on the specified percentiles.

    Parameters
    ----------
    image : np.ndarray
        3D image with shape (z, y, x).
    lower : float, optional
        Lower percentile. Default is 0.
    upper : float, optional
        Upper percentile. Default is 100.
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.

    Returns
    -------
    normalized_image : np.ndarray
        Normalized image.
    """
    image_lower = np.percentile(image, lower)
    image_upper = np.percentile(image, upper)
    normalized_image = (image - image_lower) / (image_upper - image_lower + epsilon)
    
    return normalized_image


def standardize_3d(
    image: np.ndarray,
    epsilon: float = 1e-7
) -> np.ndarray:
    """
    Standardize a 3D image to zero mean and unit variance.

    Parameters
    ----------
    image : np.ndarray
        3D image with shape (z, y, x).
    epsilon : float, optional
        Small constant for numerical stability. Default is 1e-7.

    Returns
    -------
    standardized_image : np.ndarray
        Standardized image.
    """
    image_mean = image.mean()
    image_std = image.std()
    standardized_image = (image - image_mean) / (image_std + epsilon)
    
    return standardized_image


def voronoi_transform_3d(
    coords: Sequence[np.ndarray],
    output_size: Tuple[int, int, int] = (32, 256, 256),
    dilation_iterations: int = 1,
    coords_pad_length: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform a list of 3D coordinates to generate ground truth binary labels and 
    displacement vectors from each voxel to the nearest point via a Voronoi tessellation.
    
    Adapted from 2D version for 3D data.

    Parameters
    ----------
    coords : Sequence[np.ndarray]
        List of coordinates with shape (n_spots, 3) where columns are (z, y, x).
    output_size : Tuple[int, int, int], optional
        Size of output arrays (z, y, x). Default is (32, 256, 256).
    dilation_iterations : int, optional
        Number of iterations to dilate ground truth labels. Default is 1.
    coords_pad_length : Optional[int], optional
        Padded length of the coordinates sequence. Default is None.

    Returns
    -------
    deltas : np.ndarray
        Array where each voxel is a vector (dz, dy, dx) to the nearest point in `coords`.
        Shape: (batch_size, z, y, x, 3)
    labels : np.ndarray
        Array where each voxel is a boolean for whether it contains a point in `coords`.
        Shape: (batch_size, z, y, x, 1)
    """
    
    batch_size = len(coords)
    z_size, y_size, x_size = output_size
    
    # Initialize the deltas and labels arrays
    deltas = np.zeros((batch_size, z_size, y_size, x_size, 3), dtype=float)
    labels = np.zeros((batch_size, z_size, y_size, x_size), dtype=bool)
    
    # Generate ranges for all dimensions
    z_range = np.arange(z_size)
    y_range = np.arange(y_size)
    x_range = np.arange(x_size)
    
    # Generate the 3D dilation structuring element
    structure = ndimage.generate_binary_structure(3, 1)  # 6-connectivity in 3D
    
    for k, coord in enumerate(coords):
        if len(coord) == 0:
            continue
            
        # Remove coordinates outside the output arrays
        valid_mask = (
            (coord[:, 0] > -0.5) & (coord[:, 0] < z_size - 0.5) &
            (coord[:, 1] > -0.5) & (coord[:, 1] < y_size - 0.5) &
            (coord[:, 2] > -0.5) & (coord[:, 2] < x_size - 0.5)
        )
        coord = coord[valid_mask]
        
        if len(coord) == 0:
            continue
        
        # Generate the labels array
        rounded_coords = np.rint(coord).astype(int)
        labels[k][rounded_coords[:, 0], rounded_coords[:, 1], rounded_coords[:, 2]] = True
        
        # Apply the Euclidean distance transform on the labels array
        # This gives us indices to the nearest foreground voxel
        edt_indices = ndimage.distance_transform_edt(
            ~labels[k], 
            return_distances=False, 
            return_indices=True
        )
        
        # Compute displacement vectors: for each voxel, compute vector to nearest spot
        # Create coordinate grids
        z_coords, y_coords, x_coords = np.meshgrid(z_range, y_range, x_range, indexing='ij')
        
        # Get the indices of nearest foreground voxels from EDT
        nearest_z = edt_indices[0]
        nearest_y = edt_indices[1]
        nearest_x = edt_indices[2]
        
        # For each unique nearest voxel position, find the actual coordinate
        # This is more efficient than iterating through all voxels
        unique_positions = np.unique(
            np.column_stack([nearest_z.ravel(), nearest_y.ravel(), nearest_x.ravel()]),
            axis=0
        )
        
        # Create a mapping from voxel positions to actual coordinates
        position_to_coord = {}
        for pos in unique_positions:
            nz, ny, nx = int(pos[0]), int(pos[1]), int(pos[2])
            if labels[k, nz, ny, nx]:
                # Find closest coordinate to this position
                voxel_pos = np.array([nz, ny, nx])
                distances = np.sqrt(np.sum((coord - voxel_pos) ** 2, axis=1))
                closest_idx = np.argmin(distances)
                position_to_coord[(nz, ny, nx)] = coord[closest_idx]
        
        # Compute deltas efficiently using vectorized operations
        current_pos = np.stack([z_coords, y_coords, x_coords], axis=-1)
        nearest_pos = np.stack([nearest_z, nearest_y, nearest_x], axis=-1)
        
        # For each position, get the actual coordinate from the mapping
        for z_idx in range(z_size):
            for y_idx in range(y_size):
                for x_idx in range(x_size):
                    nz, ny, nx = nearest_z[z_idx, y_idx, x_idx], \
                                 nearest_y[z_idx, y_idx, x_idx], \
                                 nearest_x[z_idx, y_idx, x_idx]
                    key = (int(nz), int(ny), int(nx))
                    if key in position_to_coord:
                        actual_coord = position_to_coord[key]
                        deltas[k, z_idx, y_idx, x_idx, :] = actual_coord - np.array([z_idx, y_idx, x_idx])
                    else:
                        # Fallback: use nearest position directly
                        deltas[k, z_idx, y_idx, x_idx, :] = np.array([nz, ny, nx]) - np.array([z_idx, y_idx, x_idx])
        
        # Dilate the labels array if necessary
        if dilation_iterations > 0:
            labels[k] = ndimage.binary_dilation(
                labels[k], 
                structure=structure, 
                iterations=dilation_iterations
            )
    
    # Expand the shape of the labels array
    labels = np.expand_dims(labels, axis=-1)
    
    return deltas, labels
