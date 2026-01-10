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
    tile_size: Tuple[int, int, int] = (16, 128, 128),
    min_spots: int = 1,
    train_size: float = 0.70,
    test_size: float = 0.15,
    overlap_factor: float = 0.0,
    batch_size: int = 1,
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
        
        # Use a fixed number of batch files per split (cycling through them) to avoid creating thousands of files
        MAX_BATCH_FILES_PER_SPLIT = 20  # Reuse same 20 files per split, cycling through them
        batch_file_handles = {
            'train': {'tiles_files': [], 'coords_files': [], 'current_idx': 0, 'counts': []},
            'test': {'tiles_files': [], 'coords_files': [], 'current_idx': 0, 'counts': []},
            'valid': {'tiles_files': [], 'coords_files': [], 'current_idx': 0, 'counts': []}
        }
        
        # Pre-create fixed batch files for each split
        for split_name in ['train', 'test', 'valid']:
            for i in range(MAX_BATCH_FILES_PER_SPLIT):
                batch_file_handles[split_name]['tiles_files'].append(
                    os.path.join(tmpdir, f'{split_name}_tiles_fixed_{i}.npy'))
                batch_file_handles[split_name]['coords_files'].append(
                    os.path.join(tmpdir, f'{split_name}_coords_fixed_{i}.npy'))
                batch_file_handles[split_name]['counts'].append(0)
        
        def flush_batch_cycled(split_name, tiles_list, coords_list, batch_file_handles):
            """Save batch to a cycled file (reuse same files)"""
            if len(tiles_list) == 0:
                return
            
            # Get current file index for this split
            idx = batch_file_handles[split_name]['current_idx']
            batch_file_tiles = batch_file_handles[split_name]['tiles_files'][idx]
            batch_file_coords = batch_file_handles[split_name]['coords_files'][idx]
            
            # Convert to arrays
            tiles_array = np.empty(len(tiles_list), dtype=object)
            coords_array = np.empty(len(coords_list), dtype=object)
            tiles_array[:] = tiles_list
            coords_array[:] = coords_list
            
            # If file exists, load and append; otherwise create new
            if os.path.exists(batch_file_tiles) and batch_file_handles[split_name]['counts'][idx] > 0:
                existing_tiles = np.load(batch_file_tiles, allow_pickle=True)
                existing_coords = np.load(batch_file_coords, allow_pickle=True)
                tiles_array = np.concatenate([existing_tiles, tiles_array])
                coords_array = np.concatenate([existing_coords, coords_array])
                del existing_tiles, existing_coords
            
            # Save
            np.save(batch_file_tiles, tiles_array, allow_pickle=True)
            np.save(batch_file_coords, coords_array, allow_pickle=True)
            batch_file_handles[split_name]['counts'][idx] += len(tiles_list)
            
            # Cycle to next file
            batch_file_handles[split_name]['current_idx'] = (idx + 1) % MAX_BATCH_FILES_PER_SPLIT
            
            # Clear lists immediately
            tiles_list.clear()
            coords_list.clear()
            del tiles_array, coords_array
            gc.collect()
        
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
            if tile_idx < train_end:
                train_tiles.append(tile)
                train_coords.append(tile_coords)
                if len(train_tiles) >= batch_size:
                    flush_batch_cycled('train', train_tiles, train_coords, batch_file_handles)
            elif tile_idx < test_end:
                test_tiles.append(tile)
                test_coords.append(tile_coords)
                if len(test_tiles) >= batch_size:
                    flush_batch_cycled('test', test_tiles, test_coords, batch_file_handles)
            else:
                valid_tiles.append(tile)
                valid_coords.append(tile_coords)
                if len(valid_tiles) >= batch_size:
                    flush_batch_cycled('valid', valid_tiles, valid_coords, batch_file_handles)
            
            # Delete tile immediately (it's been copied to list already)
            del tile, tile_coords
            # Force GC every 10 tiles to keep memory low
            if tile_idx % 10 == 0:
                gc.collect()
        
        # Flush remaining batches
        if verbose:
            print(f"\n  Flushing remaining batches...")
        if len(train_tiles) > 0:
            flush_batch_cycled('train', train_tiles, train_coords, batch_file_handles)
        if len(test_tiles) > 0:
            flush_batch_cycled('test', test_tiles, test_coords, batch_file_handles)
        if len(valid_tiles) > 0:
            flush_batch_cycled('valid', valid_tiles, valid_coords, batch_file_handles)
        
        # Collect list of non-empty batch files for each split
        batch_file_lists = {
            'train': {'tiles': [], 'coords': []},
            'test': {'tiles': [], 'coords': []},
            'valid': {'tiles': [], 'coords': []}
        }
        for split_name in ['train', 'test', 'valid']:
            for i in range(MAX_BATCH_FILES_PER_SPLIT):
                if batch_file_handles[split_name]['counts'][i] > 0:
                    batch_file_lists[split_name]['tiles'].append(batch_file_handles[split_name]['tiles_files'][i])
                    batch_file_lists[split_name]['coords'].append(batch_file_handles[split_name]['coords_files'][i])
        
        train_batch_count = len(batch_file_lists['train']['tiles'])
        test_batch_count = len(batch_file_lists['test']['tiles'])
        valid_batch_count = len(batch_file_lists['valid']['tiles'])
        
        # Ultra memory-efficient: Combine batches incrementally, save splits one at a time
        # Initialize counts
        train_count = 0
        test_count = 0
        valid_count = 0
        
        if verbose:
            print(f"\nStep 4: Combining batches with minimal memory usage...")
            print(f"  Total batch files: train={train_batch_count}, test={test_batch_count}, valid={valid_batch_count}")
        
        # Process train split - combine incrementally, one batch at a time
        if len(batch_file_lists['train']['tiles']) > 0:
            if verbose:
                print(f"  Processing train split ({train_batch_count} batches)...")
            x_train = None
            for i, batch_file in enumerate(batch_file_lists['train']['tiles']):
                arr = np.load(batch_file, allow_pickle=True)
                if x_train is None:
                    x_train = arr
                else:
                    x_train = np.concatenate([x_train, arr])
                    del arr
                if i % 5 == 0:  # More frequent GC
                    gc.collect()
            
            y_train = None
            for i, batch_file in enumerate(batch_file_lists['train']['coords']):
                arr = np.load(batch_file, allow_pickle=True)
                if y_train is None:
                    y_train = arr
                else:
                    y_train = np.concatenate([y_train, arr])
                    del arr
                if i % 5 == 0:
                    gc.collect()
            
            train_count = len(x_train)
            # Save train split immediately
            temp_train = os.path.join(tmpdir, 'train.npz')
            np.savez_compressed(temp_train, x=x_train, y=y_train)
            del x_train, y_train
            batch_file_lists['train'] = {'tiles': [], 'coords': []}
            gc.collect()
        else:
            temp_train = os.path.join(tmpdir, 'train.npz')
            np.savez_compressed(temp_train, x=np.empty(0, dtype=object), y=np.empty(0, dtype=object))
        
        # Process test split
        if len(batch_file_lists['test']['tiles']) > 0:
            if verbose:
                print(f"  Processing test split ({test_batch_count} batches)...")
            x_test = None
            for i, batch_file in enumerate(batch_file_lists['test']['tiles']):
                arr = np.load(batch_file, allow_pickle=True)
                if x_test is None:
                    x_test = arr
                else:
                    x_test = np.concatenate([x_test, arr])
                    del arr
                if i % 5 == 0:
                    gc.collect()
            
            y_test = None
            for i, batch_file in enumerate(batch_file_lists['test']['coords']):
                arr = np.load(batch_file, allow_pickle=True)
                if y_test is None:
                    y_test = arr
                else:
                    y_test = np.concatenate([y_test, arr])
                    del arr
                if i % 5 == 0:
                    gc.collect()
            
            test_count = len(x_test)
            temp_test = os.path.join(tmpdir, 'test.npz')
            np.savez_compressed(temp_test, x=x_test, y=y_test)
            del x_test, y_test
            batch_file_lists['test'] = {'tiles': [], 'coords': []}
            gc.collect()
        else:
            temp_test = os.path.join(tmpdir, 'test.npz')
            np.savez_compressed(temp_test, x=np.empty(0, dtype=object), y=np.empty(0, dtype=object))
        
        # Process validation split
        if len(batch_file_lists['valid']['tiles']) > 0:
            if verbose:
                print(f"  Processing validation split ({valid_batch_count} batches)...")
            x_valid = None
            for i, batch_file in enumerate(batch_file_lists['valid']['tiles']):
                arr = np.load(batch_file, allow_pickle=True)
                if x_valid is None:
                    x_valid = arr
                else:
                    x_valid = np.concatenate([x_valid, arr])
                    del arr
                if i % 5 == 0:
                    gc.collect()
            
            y_valid = None
            for i, batch_file in enumerate(batch_file_lists['valid']['coords']):
                arr = np.load(batch_file, allow_pickle=True)
                if y_valid is None:
                    y_valid = arr
                else:
                    y_valid = np.concatenate([y_valid, arr])
                    del arr
                if i % 5 == 0:
                    gc.collect()
            
            valid_count = len(x_valid)
            temp_valid = os.path.join(tmpdir, 'valid.npz')
            np.savez_compressed(temp_valid, x=x_valid, y=y_valid)
            del x_valid, y_valid
            batch_file_lists['valid'] = {'tiles': [], 'coords': []}
            gc.collect()
        else:
            temp_valid = os.path.join(tmpdir, 'valid.npz')
            np.savez_compressed(temp_valid, x=np.empty(0, dtype=object), y=np.empty(0, dtype=object))
        
        # Finally, load the three temp files and combine (this is the only time all are in memory)
        if verbose:
            print(f"  Creating final dataset file (loading compressed splits)...")
        train_data = np.load(temp_train, allow_pickle=True)
        test_data = np.load(temp_test, allow_pickle=True)
        valid_data = np.load(temp_valid, allow_pickle=True)
        
        # Save final dataset
        np.savez_compressed(path,
                           x_train=train_data['x'], y_train=train_data['y'],
                           x_test=test_data['x'], y_test=test_data['y'],
                           x_valid=valid_data['x'], y_valid=valid_data['y'])
        
        del train_data, test_data, valid_data
        gc.collect()
    
    if verbose:
        print(f"\n✓ Dataset saved to {path}")
        print(f"  Training: {train_count} tiles")
        print(f"  Validation: {valid_count} tiles")
        print(f"  Test: {test_count} tiles")
