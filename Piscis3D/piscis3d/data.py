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
from tifffile import imread, TiffFile
import tifffile


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
    tile_size: Tuple[int, int, int] = (8, 64, 64),
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
        
        # Load coordinates and get image shape WITHOUT loading the full image
        try:
            coords = np.load(coord_path)
            # Get image shape without loading entire image using TiffFile metadata
            with TiffFile(img_path) as tif:
                if len(tif.series) == 0:
                    if verbose:
                        print(f"\n  Warning: No series found in {img_path}, skipping")
                    continue
                series = tif.series[0]
                if series.ndim != 3:
                    if verbose:
                        print(f"\n  Warning: Image {img_idx} has {series.ndim} dims, expected 3, skipping")
                    continue
                # Get shape from metadata - this doesn't load the image
                z_max, y_max, x_max = series.shape
        except Exception as e:
            if verbose:
                print(f"\n  Warning: Failed to read {img_path}: {e}, skipping")
            continue
        
        # Validate coordinates
        if coords.ndim != 2 or coords.shape[1] != 3:
            if verbose:
                print(f"\n  Warning: Coords {img_idx} have shape {coords.shape}, expected (n, 3), skipping")
            continue
        
        # Remove duplicates
        coords = remove_duplicate_coords_3d(coords.astype(np.float32))
        
        # Count tiles for this image using only coordinates and image shape (no image loaded)
        z_size, y_size, x_size = tile_size
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
        
        # Clear coordinates from memory
        del coords
        if img_idx % 3 == 0:
            gc.collect()
    
    if verbose:
        print(f"\n  Found {total_tiles} valid tiles from {len(image_paths)} images")
    
    if total_tiles == 0:
        raise ValueError("No valid tiles found! Check your images and min_spots setting.")
    
    # Step 2: Group tiles by image first, then shuffle within each image's tiles
    # This minimizes image reloading while still providing some randomness
    if verbose:
        print(f"\nStep 2: Grouping tiles by image and preparing for extraction...")
    tiles_by_image = {}
    for tile_idx, (img_idx, z_start, y_start, x_start) in enumerate(tile_metadata):
        if img_idx not in tiles_by_image:
            tiles_by_image[img_idx] = []
        tiles_by_image[img_idx].append((tile_idx, z_start, y_start, x_start))
    
    # Shuffle tiles within each image for some randomness
    for img_idx in tiles_by_image.keys():
        img_key = random.fold_in(key, img_idx)
        perms = random.permutation(img_key, len(tiles_by_image[img_idx]))
        tiles_by_image[img_idx] = [tiles_by_image[img_idx][i] for i in perms]
    
    if verbose:
        print(f"  Tiles grouped into {len(tiles_by_image)} images")
        for img_idx in sorted(tiles_by_image.keys()):
            print(f"    Image {img_idx}: {len(tiles_by_image[img_idx])} tiles")
    
    # Step 3: Extract tiles image-by-image (process each 2GB image once, extract all its tiles)
    if verbose:
        print(f"\nStep 3: Extracting tiles image-by-image (one 2GB image at a time)...")
    
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
        
        # Track current image - use memory mapping to avoid loading full 2GB images
        current_img_idx = None
        current_tif_file = None  # Keep TiffFile handle open for metadata
        current_mmap_array = None  # Memory-mapped array (reused for all tiles from same image)
        current_image_shape = None
        current_coords = None
        current_img_path = None
        
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
        
        # Process each image separately: load once, extract all tiles, save immediately, then move to next
        # This way we never keep more than one 2GB image in memory at a time
        global_tile_counter = 0
        tile_split_assignments = []  # Will store (tile_file, split_name) for later assignment
        
        # First pass: Extract all tiles from each image and save them to individual files
        # This avoids keeping tiles in memory
        for img_idx in sorted(tiles_by_image.keys()):
            n_tiles_for_img = len(tiles_by_image[img_idx])
            if verbose:
                print(f"  Processing image {img_idx+1}/{len(tiles_by_image)}: {n_tiles_for_img} tiles...", end='\r')
            
            img_path = image_paths[img_idx]
            coord_path = coord_paths[img_idx]
            
            try:
                # Get shape from metadata (doesn't load image)
                tif_file = TiffFile(img_path)
                series = tif_file.series[0]
                image_shape = series.shape
                
                # Try memory mapping - if it works, we can slice efficiently
                mmap_array = None
                try:
                    mmap_array = tifffile.memmap(img_path)
                except Exception as e:
                    if verbose:
                        print(f"\n  Warning: memmap failed, using page-by-page: {e}")
                
                # Load coordinates (small, not a problem)
                coords = np.load(coord_path).astype(np.float32)
                coords = remove_duplicate_coords_3d(coords)
                
                # Extract and save each tile immediately (don't accumulate)
                for local_tile_idx, (z_start, y_start, x_start) in enumerate(tiles_by_image[img_idx]):
                    z_end = min(z_start + tile_size[0], image_shape[0])
                    y_end = min(y_start + tile_size[1], image_shape[1])
                    x_end = min(x_start + tile_size[2], image_shape[2])
                    
                    # Extract tile using memory mapping or page-by-page
                    if mmap_array is not None:
                        tile = np.ascontiguousarray(mmap_array[z_start:z_end, y_start:y_end, x_start:x_end]).astype(np.float32)
                    else:
                        # Page-by-page fallback
                        tile_pages = []
                        for z_page in range(z_start, z_end):
                            if z_page < len(series.pages):
                                page_data = series.pages[z_page].asarray()
                                tile_pages.append(page_data[y_start:y_end, x_start:x_end])
                        if tile_pages:
                            tile = np.stack(tile_pages, axis=0).astype(np.float32)
                        else:
                            tile = np.zeros((z_end - z_start, y_end - y_start, x_end - x_start), dtype=np.float32)
                    
                    # Pad if needed
                    if tile.shape != tile_size:
                        pad_z = tile_size[0] - tile.shape[0]
                        pad_y = tile_size[1] - tile.shape[1]
                        pad_x = tile_size[2] - tile.shape[2]
                        tile = np.pad(tile, ((0, pad_z), (0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
                    
                    # Extract coordinates
                    mask = (
                        (coords[:, 0] >= z_start) & (coords[:, 0] < z_end) &
                        (coords[:, 1] >= y_start) & (coords[:, 1] < y_end) &
                        (coords[:, 2] >= x_start) & (coords[:, 2] < x_end)
                    )
                    tile_coords = coords[mask].copy()
                    tile_coords[:, 0] -= z_start
                    tile_coords[:, 1] -= y_start
                    tile_coords[:, 2] -= x_start
                    
                    # Save tile immediately to a temporary file (one file per tile)
                    tile_file = os.path.join(tmpdir, f'tile_{global_tile_counter}.npy')
                    coord_file = os.path.join(tmpdir, f'coords_{global_tile_counter}.npy')
                    np.save(tile_file, tile)
                    np.save(coord_file, tile_coords)
                    
                    # Store file paths (original_idx not needed since we'll shuffle file paths)
                    tile_split_assignments.append((tile_file, coord_file))
                    
                    global_tile_counter += 1
                    del tile, tile_coords
                    
                    # GC every 10 tiles
                    if local_tile_idx % 10 == 0:
                        gc.collect()
                
                # Close image resources immediately
                if mmap_array is not None:
                    del mmap_array
                tif_file.close()
                del coords
                gc.collect()
                
            except Exception as e:
                if verbose:
                    print(f"\n  Error processing image {img_idx}: {e}, skipping")
                import traceback
                traceback.print_exc()
                continue
        
        if verbose:
            print(f"\n  Extracted {global_tile_counter} tiles total, saved to individual files")
        
        # Shuffle tile file assignments
        if verbose:
            print(f"\nStep 4: Shuffling {global_tile_counter} tile files...")
        shuffle_key = random.split(key, 1)[0]
        perms = np.asarray(random.permutation(shuffle_key, len(tile_split_assignments)))
        tile_split_assignments_shuffled = [tile_split_assignments[i] for i in perms]
        del tile_split_assignments
        gc.collect()
        
        # Calculate splits
        split_indices = np.rint(np.cumsum((train_size, test_size)) * global_tile_counter).astype(int)
        train_end = split_indices[0]
        test_end = split_indices[1]
        
        if verbose:
            print(f"  Train: 0-{train_end} ({train_end} tiles)")
            print(f"  Test: {train_end}-{test_end} ({test_end - train_end} tiles)")
            print(f"  Valid: {test_end}-{global_tile_counter} ({global_tile_counter - test_end} tiles)")
        
        # Step 5: Load tiles from files and assign to splits (load one at a time)
        if verbose:
            print(f"\nStep 5: Loading tiles from files and assigning to splits...")
        
        for tile_idx, (tile_file, coord_file) in enumerate(tile_split_assignments_shuffled):
            if verbose and tile_idx % 500 == 0:
                print(f"  Loading tile {tile_idx+1}/{global_tile_counter}...", end='\r')
            
            # Load tile from file (only one tile in memory at a time)
            tile = np.load(tile_file)
            tile_coords = np.load(coord_file)
            
            # Assign to split and flush immediately
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
            
            del tile, tile_coords
            if tile_idx % 10 == 0:
                gc.collect()
        
        del tile_split_assignments_shuffled
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
            print(f"\nStep 6: Combining batches with minimal memory usage...")
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
