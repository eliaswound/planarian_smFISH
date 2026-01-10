"""
3D Dataset generation for Piscis.
Handles 3D images (z, y, x) and 3D coordinates (z, y, x).
"""

import jax
import numpy as np
from jax import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from piscis3d.utils import remove_duplicate_coords_3d


def generate_3d_tiles(
    image: np.ndarray,
    coords: np.ndarray,
    tile_size: Tuple[int, int, int] = (32, 256, 256),
    min_spots: int = 1
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate 3D tiles from a 3D image with associated coordinates.
    
    Parameters
    ----------
    image : np.ndarray
        3D image with shape (z, y, x)
    coords : np.ndarray
        Coordinates with shape (n_spots, 3) where columns are (z, y, x)
    tile_size : Tuple[int, int, int]
        Size of each tile (z, y, x)
    min_spots : int
        Minimum number of spots per tile
        
    Returns
    -------
    tiles : List[np.ndarray]
        List of 3D image tiles
    tile_coords : List[np.ndarray]
        List of coordinate arrays for each tile
    """
    z_size, y_size, x_size = tile_size
    z_max, y_max, x_max = image.shape
    
    tiles = []
    tile_coords = []
    
    # Generate tiles with overlap
    # Step size can be adjusted for more/less overlap
    z_step = max(1, z_size // 2)
    y_step = max(1, y_size // 2)
    x_step = max(1, x_size // 2)
    
    for z_start in range(0, z_max, z_step):
        for y_start in range(0, y_max, y_step):
            for x_start in range(0, x_max, x_step):
                z_end = min(z_start + z_size, z_max)
                y_end = min(y_start + y_size, y_max)
                x_end = min(x_start + x_size, x_max)
                
                # Extract tile
                tile = image[z_start:z_end, y_start:y_end, x_start:x_end]
                
                # Pad tile if smaller than tile_size
                if tile.shape != tile_size:
                    pad_z = z_size - tile.shape[0]
                    pad_y = y_size - tile.shape[1]
                    pad_x = x_size - tile.shape[2]
                    tile = np.pad(tile, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
                
                # Find coordinates within this tile
                # Coordinates are in (z, y, x) format
                mask = (
                    (coords[:, 0] >= z_start) & (coords[:, 0] < z_end) &
                    (coords[:, 1] >= y_start) & (coords[:, 1] < y_end) &
                    (coords[:, 2] >= x_start) & (coords[:, 2] < x_end)
                )
                tile_coords_subset = coords[mask].copy()
                
                # Adjust coordinates to tile-local coordinates
                if len(tile_coords_subset) > 0:
                    tile_coords_subset[:, 0] -= z_start
                    tile_coords_subset[:, 1] -= y_start
                    tile_coords_subset[:, 2] -= x_start
                
                # Only keep tiles with enough spots
                if len(tile_coords_subset) >= min_spots:
                    tiles.append(tile)
                    tile_coords.append(tile_coords_subset)
    
    return tiles, tile_coords


def generate_dataset_3d(
    path: str,
    images: List[np.ndarray],
    coords: List[np.ndarray],
    key: jax.Array,
    tile_size: Tuple[int, int, int] = (32, 256, 256),
    min_spots: int = 1,
    train_size: float = 0.70,
    test_size: float = 0.15
) -> None:
    """
    Generate a 3D dataset from images and spot coordinates.

    Parameters
    ----------
    path : str
        Path to save dataset.
    images : List[np.ndarray]
        List of 3D images with shape (z, y, x).
    coords : List[np.ndarray]
        List of ground truth spot coordinates with shape (n_spots, 3) where columns are (z, y, x).
    key : jax.Array
        Random key used for splitting the dataset into training, validation, and test sets.
    tile_size : Tuple[int, int, int], optional
        Tile size used for splitting images (z, y, x). Default is (32, 256, 256).
    min_spots : int, optional
        Minimum number of spots per tile. Default is 1.
    train_size : float, optional
        Fraction of dataset used for training. Default is 0.70.
    test_size : float, optional
        Fraction of dataset used for testing. Default is 0.15.
    """
    
    # Remove duplicate coordinates.
    for i in range(len(coords)):
        coords[i] = remove_duplicate_coords_3d(coords[i])
    
    tiled_images_list = []
    tiled_coords_list = []
    
    print(f"Generating 3D tiles from {len(images)} images...")
    for idx, (image, c) in enumerate(zip(images, coords)):
        if idx % 10 == 0:
            print(f"  Processing image {idx+1}/{len(images)}...")
        
        # Validate image dimensions
        if image.ndim != 3:
            raise ValueError(f"Image {idx} has {image.ndim} dimensions, expected 3 (z, y, x)")
        
        # Validate coordinate dimensions
        if c.ndim != 2 or c.shape[1] != 3:
            raise ValueError(f"Coordinates for image {idx} have shape {c.shape}, expected (n_spots, 3)")
        
        # Generate 3D tiles
        tiles, tile_coords = generate_3d_tiles(image, c, tile_size, min_spots)
        
        tiled_images_list.extend(tiles)
        tiled_coords_list.extend(tile_coords)
    
    print(f"Generated {len(tiled_images_list)} tiles total")
    
    # Convert to numpy arrays
    tiled_images = np.empty(len(tiled_images_list), dtype=object)
    tiled_coords = np.empty(len(tiled_coords_list), dtype=object)
    tiled_images[:] = tiled_images_list
    tiled_coords[:] = tiled_coords_list
    
    # Randomly shuffle the tiles.
    size = len(tiled_images)
    perms = np.asarray(random.permutation(key, size))
    tiled_images = tiled_images[perms]
    tiled_coords = tiled_coords[perms]
    
    # Split the dataset into training, validation, and test sets.
    split_indices = np.rint(np.cumsum((train_size, test_size)) * size).astype(int)
    x_train = tiled_images[:split_indices[0]]
    y_train = tiled_coords[:split_indices[0]]
    x_valid = tiled_images[split_indices[1]:]
    y_valid = tiled_coords[split_indices[1]:]
    x_test = tiled_images[split_indices[0]:split_indices[1]]
    y_test = tiled_coords[split_indices[0]:split_indices[1]]
    
    # Create the dataset dictionary.
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, x_test=x_test, y_test=y_test)
    
    print(f"Dataset saved to {path}")
    print(f"  Training: {len(x_train)} tiles")
    print(f"  Validation: {len(x_valid)} tiles")
    print(f"  Test: {len(x_test)} tiles")


def load_datasets_3d(
    path: str,
    adjustment: Optional[str] = 'standardize',
    load_train: bool = True,
    load_valid: bool = True,
    load_test: bool = True
) -> Dict:
    """
    Load 3D datasets from a directory or file.

    Parameters
    ----------
    path : str
        Path to a dataset or directory of datasets.
    adjustment : Optional[str], optional
        Adjustment type applied to images. Supported types are 'normalize' and 'standardize'. Default is 'standardize'.
    load_train : bool, optional
        Whether to load the training set. Default is True.
    load_valid : bool, optional
        Whether to load the validation set. Default is True.
    load_test : bool, optional
        Whether to load the test set. Default is True.

    Returns
    -------
    dataset : Dict
        Dataset dictionary with keys 'train', 'valid', 'test' (if loaded).
    """
    from piscis3d.transforms import batch_adjust_3d
    
    # Create empty dictionaries.
    train = {'images': [], 'coords': []}
    valid = {'images': [], 'coords': []}
    test = {'images': [], 'coords': []}
    dataset = {}
    
    # Get dataset paths.
    path = Path(path)
    if path.is_file() and path.suffix == '.npz':
        dataset_paths = [path]
    else:
        dataset_paths = list(path.glob('*.npz'))
    
    # Load datasets.
    for dataset_path in dataset_paths:
        npz = np.load(dataset_path, allow_pickle=True)
        if load_train:
            train['images'].append(npz['x_train'])
            train['coords'].append(npz['y_train'])
        if load_valid:
            valid['images'].append(npz['x_valid'])
            valid['coords'].append(npz['y_valid'])
        if load_test:
            test['images'].append(npz['x_test'])
            test['coords'].append(npz['y_test'])
    
    # Combine datasets and adjust images if necessary.
    if load_train and len(train['images']) > 0:
        train['images'] = np.concatenate(train['images'])
        train['images'] = batch_adjust_3d(train['images'], adjustment)
        train['coords'] = np.concatenate(train['coords'])
        dataset['train'] = train
    if load_valid and len(valid['images']) > 0:
        valid['images'] = np.concatenate(valid['images'])
        valid['images'] = batch_adjust_3d(valid['images'], adjustment)
        valid['coords'] = np.concatenate(valid['coords'])
        dataset['valid'] = valid
    if load_test and len(test['images']) > 0:
        test['images'] = np.concatenate(test['images'])
        test['images'] = batch_adjust_3d(test['images'], adjustment)
        test['coords'] = np.concatenate(test['coords'])
        dataset['test'] = test
    
    return dataset
