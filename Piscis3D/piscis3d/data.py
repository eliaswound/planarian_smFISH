"""
3D Dataset generation for Piscis.
Handles 3D images (z, y, x) and 3D coordinates (z, y, x).

Memory-optimized version that processes images incrementally.
"""

import jax
import numpy as np
from jax import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tempfile
import os
import gc

from piscis3d.utils import remove_duplicate_coords_3d
from tifffile import imread


def generate_3d_tiles_generator(
    image: np.ndarray,
    coords: np.ndarray,
    tile_size: Tuple[int, int, int] = (32, 256, 256),
    min_spots: int = 1,
    overlap_factor: float = 0.5
):
    """
    Generator that yields 3D tiles one at a time to save memory.
    
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
    overlap_factor : float
        Overlap factor (0.0 = no overlap, 0.5 = 50% overlap). Default 0.5.
        Lower values generate fewer tiles and use less memory.
        
    Yields
    ------
    tile : np.ndarray
        3D image tile
    tile_coords : np.ndarray
        Coordinate array for the tile
    """
    z_size, y_size, x_size = tile_size
    z_max, y_max, x_max = image.shape
    
    # Reduce overlap to generate fewer tiles (saves memory)
    # Use larger step sizes based on overlap_factor
    z_step = max(1, int(z_size * (1 - overlap_factor)))
    y_step = max(1, int(y_size * (1 - overlap_factor)))
    x_step = max(1, int(x_size * (1 - overlap_factor)))
    
    for z_start in range(0, z_max, z_step):
        for y_start in range(0, y_max, y_step):
            for x_start in range(0, x_max, x_step):
                z_end = min(z_start + z_size, z_max)
                y_end = min(y_start + y_size, y_max)
                x_end = min(x_start + x_size, x_max)
                
                # Extract tile (creates a view, not a copy initially)
                tile = np.ascontiguousarray(image[z_start:z_end, y_start:y_end, x_start:x_end])
                
                # Pad tile if smaller than tile_size
                if tile.shape != tile_size:
                    pad_z = z_size - tile.shape[0]
                    pad_y = y_size - tile.shape[1]
                    pad_x = x_size - tile.shape[2]
                    tile = np.pad(tile, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
                
                # Find coordinates within this tile (vectorized)
                mask = (
                    (coords[:, 0] >= z_start) & (coords[:, 0] < z_end) &
                    (coords[:, 1] >= y_start) & (coords[:, 1] < y_end) &
                    (coords[:, 2] >= x_start) & (coords[:, 2] < x_end)
                )
                
                if np.sum(mask) >= min_spots:
                    tile_coords_subset = coords[mask].copy()
                    # Adjust coordinates to tile-local coordinates
                    tile_coords_subset[:, 0] -= z_start
                    tile_coords_subset[:, 1] -= y_start
                    tile_coords_subset[:, 2] -= x_start
                    yield tile, tile_coords_subset


# Old function removed - use generate_dataset_3d_from_paths instead for memory efficiency


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


def generate_dataset_3d_from_paths(
    path: str,
    image_paths: List[str],
    coord_paths: List[str],
    key: jax.Array,
    tile_size: Tuple[int, int, int] = (32, 256, 256),
    min_spots: int = 1,
    train_size: float = 0.70,
    test_size: float = 0.15,
    overlap_factor: float = 0.0,
    batch_size: int = 10,
    verbose: bool = True
) -> None:
    """
    Generate a 3D dataset from image and coordinate file paths.
    Memory-optimized: loads images one at a time from disk.

    Parameters
    ----------
    path : str
        Path to save dataset.
    image_paths : List[str]
        List of paths to 3D image files.
    coord_paths : List[str]
        List of paths to coordinate files (numpy arrays).
    key : jax.Array
        Random key used for splitting the dataset.
    tile_size : Tuple[int, int, int], optional
        Tile size (z, y, x). Default is (32, 256, 256).
    min_spots : int, optional
        Minimum number of spots per tile. Default is 1.
    train_size : float, optional
        Fraction for training. Default is 0.70.
    test_size : float, optional
        Fraction for testing. Default is 0.15.
    overlap_factor : float, optional
        Overlap factor (0.0 = no overlap). Lower = fewer tiles = less memory. Default is 0.1.
    batch_size : int, optional
        Tiles per batch before writing to disk. Default is 200.
    verbose : bool, optional
        Print progress. Default is True.
    """
    if verbose:
        print(f"\n{'='*60}")
        print("Memory-optimized 3D dataset generation from file paths")
        print(f"{'='*60}")
        print(f"  Processing {len(image_paths)} images")
        print(f"  Tile size (z, y, x): {tile_size}")
        print(f"  Overlap factor: {overlap_factor} (lower = fewer tiles)")
        print(f"  Batch size: {batch_size} tiles")
        print(f"  Min spots per tile: {min_spots}")
    
    # Step 1: Count tiles by scanning through images one at a time
    if verbose:
        print(f"\nStep 1: Scanning images to count valid tiles...")
    total_tiles = 0
    tile_metadata = []  # (image_idx, z_start, y_start, x_start)
    
    for img_idx, (img_path, coord_path) in enumerate(zip(image_paths, coord_paths)):
        if verbose and img_idx % 2 == 0:
            print(f"  Scanning image {img_idx+1}/{len(image_paths)}...", end='\r')
        
        # Load only this image and coordinates (process one at a time)
        try:
            image = imread(img_path)
            coords = np.load(coord_path)
        except Exception as e:
            if verbose:
                print(f"\n  Warning: Failed to load {img_path}: {e}, skipping")
            continue
        
        # Validate
        if image.ndim != 3:
            if verbose:
                print(f"\n  Warning: Image {img_idx} has {image.ndim} dims, expected 3, skipping")
            continue
        if coords.ndim != 2 or coords.shape[1] != 3:
            if verbose:
                print(f"\n  Warning: Coords {img_idx} have shape {coords.shape}, expected (n, 3), skipping")
            continue
        
        # Remove duplicates
        coords = remove_duplicate_coords_3d(coords.astype(np.float32))
        
        # Count tiles for this image (without storing the image)
        z_size, y_size, x_size = tile_size
        z_max, y_max, x_max = image.shape
        z_step = max(1, int(z_size * (1 - overlap_factor)))
        y_step = max(1, int(y_size * (1 - overlap_factor)))
        x_step = max(1, int(x_size * (1 - overlap_factor)))
        
        for z_start in range(0, z_max, z_step):
            for y_start in range(0, y_max, y_step):
                for x_start in range(0, x_max, x_step):
                    z_end = min(z_start + z_size, z_max)
                    y_end = min(y_start + y_size, y_max)
                    x_end = min(x_start + x_size, x_max)
                    
                    # Check spots without extracting tile
                    mask = (
                        (coords[:, 0] >= z_start) & (coords[:, 0] < z_end) &
                        (coords[:, 1] >= y_start) & (coords[:, 1] < y_end) &
                        (coords[:, 2] >= x_start) & (coords[:, 2] < x_end)
                    )
                    
                    if np.sum(mask) >= min_spots:
                        tile_metadata.append((img_idx, z_start, y_start, x_start))
                        total_tiles += 1
        
        # Clear image from memory immediately and force garbage collection
        del image, coords
        if img_idx % 3 == 0:  # Force GC every few images to free memory aggressively
            gc.collect()
    
    if verbose:
        print(f"\n  Found {total_tiles} valid tiles from {len(image_paths)} images")
    
    if total_tiles == 0:
        raise ValueError("No valid tiles found! Check your images and min_spots setting.")
    
    # Step 2: Shuffle tile indices
    if verbose:
        print(f"\nStep 2: Shuffling {total_tiles} tiles...")
    perms = np.asarray(random.permutation(key, total_tiles))
    tile_metadata_shuffled = [tile_metadata[i] for i in perms]
    
    # Calculate splits
    split_indices = np.rint(np.cumsum((train_size, test_size)) * total_tiles).astype(int)
    train_end = split_indices[0]
    test_end = split_indices[1]
    
    if verbose:
        print(f"  Train: 0-{train_end} ({train_end} tiles)")
        print(f"  Test: {train_end}-{test_end} ({test_end - train_end} tiles)")
        print(f"  Valid: {test_end}-{total_tiles} ({total_tiles - test_end} tiles)")
    
    # Step 3: Extract tiles incrementally and save in batches
    if verbose:
        print(f"\nStep 3: Extracting tiles and saving in batches...")
    
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use temporary directory for intermediate files
    with tempfile.TemporaryDirectory() as tmpdir:
        train_tiles_file = os.path.join(tmpdir, 'train_tiles.npy')
        train_coords_file = os.path.join(tmpdir, 'train_coords.npy')
        test_tiles_file = os.path.join(tmpdir, 'test_tiles.npy')
        test_coords_file = os.path.join(tmpdir, 'test_coords.npy')
        valid_tiles_file = os.path.join(tmpdir, 'valid_tiles.npy')
        valid_coords_file = os.path.join(tmpdir, 'valid_coords.npy')
        
        # Track current image in memory (load only when needed)
        current_img_idx = None
        current_image = None
        current_coords = None
        
        # Batch accumulators and counters
        train_tiles = []
        train_coords = []
        test_tiles = []
        test_coords = []
        valid_tiles = []
        valid_coords = []
        
        train_batch_count = 0
        test_batch_count = 0
        valid_batch_count = 0
        
        batch_file_lists = {
            'train': {'tiles': [], 'coords': []},
            'test': {'tiles': [], 'coords': []},
            'valid': {'tiles': [], 'coords': []}
        }
        
        def flush_batch(split_name, tiles_list, coords_list, batch_counter, batch_file_lists):
            """Save batch to temporary file immediately and track file paths"""
            if len(tiles_list) == 0:
                return
            
            # Create unique batch file names
            batch_file_tiles = os.path.join(tmpdir, f'{split_name}_tiles_batch_{batch_counter}.npy')
            batch_file_coords = os.path.join(tmpdir, f'{split_name}_coords_batch_{batch_counter}.npy')
            
            # Convert to arrays and save immediately
            tiles_array = np.empty(len(tiles_list), dtype=object)
            coords_array = np.empty(len(coords_list), dtype=object)
            tiles_array[:] = tiles_list
            coords_array[:] = coords_list
            
            # Save immediately to free memory
            np.save(batch_file_tiles, tiles_array, allow_pickle=True)
            np.save(batch_file_coords, coords_array, allow_pickle=True)
            
            # Track file paths
            batch_file_lists[split_name]['tiles'].append(batch_file_tiles)
            batch_file_lists[split_name]['coords'].append(batch_file_coords)
            
            # Clear lists and arrays immediately to free memory
            tiles_list.clear()
            coords_list.clear()
            del tiles_array, coords_array
            gc.collect()  # Aggressive cleanup after each batch
        
        # Process each tile
        for tile_idx, (img_idx, z_start, y_start, x_start) in enumerate(tile_metadata_shuffled):
            if verbose and tile_idx % 500 == 0:
                print(f"  Processing tile {tile_idx+1}/{total_tiles}...", end='\r')
            
            # Load image if not already loaded
            if current_img_idx != img_idx:
                # Explicitly free previous image from memory
                if current_image is not None:
                    del current_image
                if current_coords is not None:
                    del current_coords
                current_image = None
                current_coords = None
                gc.collect()  # Force garbage collection
                
                # Load new image
                img_path = image_paths[img_idx]
                coord_path = coord_paths[img_idx]
                current_image = imread(img_path)
                current_coords = np.load(coord_path).astype(np.float32)
                current_coords = remove_duplicate_coords_3d(current_coords)
                current_img_idx = img_idx
            
            # Extract tile - convert to float32 immediately to reduce memory
            z_end = min(z_start + tile_size[0], current_image.shape[0])
            y_end = min(y_start + tile_size[1], current_image.shape[1])
            x_end = min(x_start + tile_size[2], current_image.shape[2])
            
            # Extract and convert to float32 to reduce memory footprint
            tile = np.ascontiguousarray(current_image[z_start:z_end, y_start:y_end, x_start:x_end]).astype(np.float32)
            if tile.shape != tile_size:
                pad_z = tile_size[0] - tile.shape[0]
                pad_y = tile_size[1] - tile.shape[1]
                pad_x = tile_size[2] - tile.shape[2]
                tile = np.pad(tile, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
            
            # Extract and adjust coordinates
            mask = (
                (current_coords[:, 0] >= z_start) & (current_coords[:, 0] < z_end) &
                (current_coords[:, 1] >= y_start) & (current_coords[:, 1] < y_end) &
                (current_coords[:, 2] >= x_start) & (current_coords[:, 2] < x_end)
            )
            tile_coords = current_coords[mask].copy()
            tile_coords[:, 0] -= z_start
            tile_coords[:, 1] -= y_start
            tile_coords[:, 2] -= x_start
            
            # Add to appropriate split and flush immediately if batch is full
            # Use very small batch size to minimize memory
            if tile_idx < train_end:
                train_tiles.append(tile)
                train_coords.append(tile_coords)
                # Flush immediately when batch reaches size (don't wait)
                if len(train_tiles) >= batch_size:
                    flush_batch('train', train_tiles, train_coords, train_batch_count, batch_file_lists)
                    train_batch_count += 1
            elif tile_idx < test_end:
                test_tiles.append(tile)
                test_coords.append(tile_coords)
                if len(test_tiles) >= batch_size:
                    flush_batch('test', test_tiles, test_coords, test_batch_count, batch_file_lists)
                    test_batch_count += 1
            else:
                valid_tiles.append(tile)
                valid_coords.append(tile_coords)
                if len(valid_tiles) >= batch_size:
                    flush_batch('valid', valid_tiles, valid_coords, valid_batch_count, batch_file_lists)
                    valid_batch_count += 1
            
            # Delete tile immediately (it's been copied to list already)
            del tile, tile_coords
            # Force GC every 10 tiles to keep memory low
            if tile_idx % 10 == 0:
                gc.collect()
        
        # Flush remaining batches
        if verbose:
            print(f"\n  Flushing remaining batches...")
        if len(train_tiles) > 0:
            flush_batch('train', train_tiles, train_coords, train_batch_count, batch_file_lists)
        if len(test_tiles) > 0:
            flush_batch('test', test_tiles, test_coords, test_batch_count, batch_file_lists)
        if len(valid_tiles) > 0:
            flush_batch('valid', valid_tiles, valid_coords, valid_batch_count, batch_file_lists)
        
        # Combine batch files efficiently
        if verbose:
            print(f"\nStep 4: Combining {train_batch_count + test_batch_count + valid_batch_count} batches into final dataset...")
        
        def combine_batch_files(batch_file_list):
            """Combine batch files efficiently with minimal memory usage - write directly to final file"""
            if len(batch_file_list) == 0:
                return np.empty(0, dtype=object)
            
            if len(batch_file_list) == 1:
                # Single batch - load and return directly
                result = np.load(batch_file_list[0], allow_pickle=True)
                return result
            
            # For multiple batches, load and combine in smaller chunks to avoid memory spikes
            # Load first batch
            result = np.load(batch_file_list[0], allow_pickle=True)
            
            # Combine remaining batches one at a time
            for i, batch_file in enumerate(batch_file_list[1:], 1):
                arr = np.load(batch_file, allow_pickle=True)
                # Concatenate incrementally
                result = np.concatenate([result, arr])
                del arr
                # Force garbage collection after every batch to free memory immediately
                gc.collect()
            
            return result
        
        # Combine and save each split separately to minimize peak memory
        # Save each split to temporary file, then combine files (avoids keeping all in memory)
        if verbose:
            print(f"\nStep 4: Combining batches and creating final dataset...")
            print(f"  Processing splits one at a time to minimize memory...")
        
        temp_train_file = os.path.join(tmpdir, 'temp_train.npz')
        temp_test_file = os.path.join(tmpdir, 'temp_test.npz')
        temp_valid_file = os.path.join(tmpdir, 'temp_valid.npz')
        
        # Process and save train split
        if verbose:
            print(f"  Combining and saving train split ({train_batch_count} batches)...")
        if len(batch_file_lists['train']['tiles']) > 0:
            x_train = combine_batch_files(batch_file_lists['train']['tiles'])
            y_train = combine_batch_files(batch_file_lists['train']['coords'])
            np.savez_compressed(temp_train_file, x=x_train, y=y_train)
            del x_train, y_train
            batch_file_lists['train'] = {'tiles': [], 'coords': []}
        else:
            # Empty split
            np.savez_compressed(temp_train_file, x=np.empty(0, dtype=object), y=np.empty(0, dtype=object))
        gc.collect()
        
        # Process and save test split
        if verbose:
            print(f"  Combining and saving test split ({test_batch_count} batches)...")
        if len(batch_file_lists['test']['tiles']) > 0:
            x_test = combine_batch_files(batch_file_lists['test']['tiles'])
            y_test = combine_batch_files(batch_file_lists['test']['coords'])
            np.savez_compressed(temp_test_file, x=x_test, y=y_test)
            del x_test, y_test
            batch_file_lists['test'] = {'tiles': [], 'coords': []}
        else:
            np.savez_compressed(temp_test_file, x=np.empty(0, dtype=object), y=np.empty(0, dtype=object))
        gc.collect()
        
        # Process and save validation split
        if verbose:
            print(f"  Combining and saving validation split ({valid_batch_count} batches)...")
        if len(batch_file_lists['valid']['tiles']) > 0:
            x_valid = combine_batch_files(batch_file_lists['valid']['tiles'])
            y_valid = combine_batch_files(batch_file_lists['valid']['coords'])
            np.savez_compressed(temp_valid_file, x=x_valid, y=y_valid)
            del x_valid, y_valid
            batch_file_lists['valid'] = {'tiles': [], 'coords': []}
        else:
            np.savez_compressed(temp_valid_file, x=np.empty(0, dtype=object), y=np.empty(0, dtype=object))
        gc.collect()
        
        # Load each split from temp files and combine into final dataset
        # This way we only have one split in memory at a time
        if verbose:
            print(f"  Combining splits into final dataset file...")
        train_data = np.load(temp_train_file, allow_pickle=True)
        test_data = np.load(temp_test_file, allow_pickle=True)
        valid_data = np.load(temp_valid_file, allow_pickle=True)
        
        # Save final combined dataset
        np.savez_compressed(path, 
                           x_train=train_data['x'], y_train=train_data['y'],
                           x_test=test_data['x'], y_test=test_data['y'],
                           x_valid=valid_data['x'], y_valid=valid_data['y'])
        
        # Free memory
        del train_data, test_data, valid_data
        gc.collect()
    
    if verbose:
        # Reload just to get counts for final message
        final_data = np.load(path, allow_pickle=True)
        n_train = len(final_data['x_train'])
        n_valid = len(final_data['x_valid'])
        n_test = len(final_data['x_test'])
        del final_data
        print(f"\n✓ Dataset saved to {path}")
        print(f"  Training: {n_train} tiles")
        print(f"  Validation: {n_valid} tiles")
        print(f"  Test: {n_test} tiles")
