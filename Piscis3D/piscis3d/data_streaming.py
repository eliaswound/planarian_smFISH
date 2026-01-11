"""
Streaming dataset generation for 3D Piscis - memory efficient version.
Instead of creating one large npz file, creates a dataset that can be loaded incrementally.
"""

import jax
import numpy as np
from jax import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator
import tempfile
import os
import gc
import json
from tifffile import TiffFile, memmap
import sys

from piscis3d.utils import remove_duplicate_coords_3d

# Try to import step profiler
try:
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "tests"))
    from step_memory_profiler import get_profiler
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False
    def get_profiler():
        return None


def generate_dataset_3d_streaming(
    output_dir: str,
    image_paths: List[str],
    coord_paths: List[str],
    key: jax.Array,
    tile_size: Tuple[int, int, int] = (16, 128, 128),
    min_spots: int = 1,
    train_size: float = 0.70,
    test_size: float = 0.15,
    overlap_factor: float = 0.0,
    max_tiles_per_image: int = 1000,  # Limit tiles per image to reduce memory
    verbose: bool = True
) -> str:
    """
    Generate a streaming dataset that saves tiles in batches without loading everything.
    
    Returns the path to the dataset directory containing:
    - metadata.json: Dataset metadata
    - train/, test/, valid/ directories with batch files
    - Each batch file contains up to 500 tiles (reduced for memory efficiency with large images)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create split directories
    for split in ['train', 'test', 'valid']:
        (output_dir / split).mkdir(exist_ok=True)
    
    # Get profiler if available
    profiler = get_profiler() if HAS_PROFILER else None
    
    # Step 1: Count valid tiles (without loading images)
    step_name = "Step 1: Count valid tiles"
    if profiler:
        step_context = profiler.step(step_name, "generate_dataset_3d_streaming:Step1")
        step_context.__enter__()
    else:
        step_context = None
    
    try:
        if verbose:
            print(f"\nStep 1: Scanning images to count valid tiles...")
        tile_metadata = []
        
        for img_idx, (img_path, coord_path) in enumerate(zip(image_paths, coord_paths)):
            if verbose and img_idx % 2 == 0:
                print(f"  Scanning image {img_idx+1}/{len(image_paths)}...", end='\r')
            
            try:
                # Get image shape without loading
                with TiffFile(img_path) as tif:
                    series = tif.series[0]
                    if series.ndim != 3:
                        continue
                    z_max, y_max, x_max = series.shape
                
                # Load only coordinates
                coords = np.load(coord_path).astype(np.float32)
                coords = remove_duplicate_coords_3d(coords)
                
                if coords.ndim != 2 or coords.shape[1] != 3:
                    continue
                
                # Count tiles
                z_size, y_size, x_size = tile_size
                z_step = max(1, int(z_size * (1 - overlap_factor)))
                y_step = max(1, int(y_size * (1 - overlap_factor)))
                x_step = max(1, int(x_size * (1 - overlap_factor)))
                
                image_tiles = []
                for z_start in range(0, z_max, z_step):
                    for y_start in range(0, y_max, y_step):
                        for x_start in range(0, x_max, x_step):
                            z_end = min(z_start + z_size, z_max)
                            y_end = min(y_start + y_size, y_max)
                            x_end = min(x_start + x_size, x_max)
                            
                            mask = (
                                (coords[:, 0] >= z_start) & (coords[:, 0] < z_end) &
                                (coords[:, 1] >= y_start) & (coords[:, 1] < y_end) &
                                (coords[:, 2] >= x_start) & (coords[:, 2] < x_end)
                            )
                            
                            if np.sum(mask) >= min_spots:
                                image_tiles.append((img_idx, z_start, y_start, x_start))
                
                # Limit tiles per image
                if len(image_tiles) > max_tiles_per_image:
                    # Randomly sample tiles
                    perms = np.asarray(random.permutation(random.fold_in(key, img_idx), len(image_tiles)))
                    image_tiles = [image_tiles[i] for i in perms[:max_tiles_per_image]]
                
                tile_metadata.extend(image_tiles)
                del coords
                gc.collect()
                
            except Exception as e:
                if verbose:
                    print(f"\n  Warning: Failed to process {img_path}: {e}")
                continue
        
        total_tiles = len(tile_metadata)
        if verbose:
            print(f"\n  Found {total_tiles} valid tiles from {len(image_paths)} images")
        
        if total_tiles == 0:
            raise ValueError("No valid tiles found!")
    finally:
        if step_context:
            step_context.__exit__(None, None, None)
    
    # Step 2: Shuffle and split
    step_name = "Step 2: Shuffle and split"
    if profiler:
        step_context = profiler.step(step_name, "generate_dataset_3d_streaming:Step2")
        step_context.__enter__()
    else:
        step_context = None
    
    try:
        if verbose:
            print(f"\nStep 2: Shuffling and splitting {total_tiles} tiles...")
        perms = np.asarray(random.permutation(key, total_tiles))
        tile_metadata_shuffled = [tile_metadata[i] for i in perms]
        del tile_metadata  # Free memory after shuffling
        gc.collect()
        
        split_indices = np.rint(np.cumsum((train_size, test_size)) * total_tiles).astype(int)
        train_end = split_indices[0]
        test_end = split_indices[1]
        
        splits = {
            'train': tile_metadata_shuffled[:train_end],
            'test': tile_metadata_shuffled[train_end:test_end],
            'valid': tile_metadata_shuffled[test_end:]
        }
        del tile_metadata_shuffled  # Free after splitting
        gc.collect()
        
        if verbose:
            print(f"  Train: {len(splits['train'])} tiles")
            print(f"  Test: {len(splits['test'])} tiles")
            print(f"  Valid: {len(splits['valid'])} tiles")
    finally:
        if step_context:
            step_context.__exit__(None, None, None)
    
    # Step 3: Process each split and save in batches
    # Reduced batch size for 2GB images to prevent memory spikes
    BATCH_SIZE = 500  # Tiles per batch file (reduced from 1000 for memory efficiency)
    
    metadata = {
        'tile_size': tile_size,
        'total_tiles': total_tiles,
        'splits': {},
        'image_paths': image_paths,
        'coord_paths': coord_paths
    }
    
    for split_name, tiles in splits.items():
        step_name = f"Step 3: Process {split_name} split"
        if profiler:
            step_context = profiler.step(step_name, f"generate_dataset_3d_streaming:Step3:{split_name}")
            step_context.__enter__()
        else:
            step_context = None
        
        try:
            if verbose:
                print(f"\nStep 3: Processing {split_name} split ({len(tiles)} tiles)...")
            
            split_dir = output_dir / split_name
            n_batches = (len(tiles) + BATCH_SIZE - 1) // BATCH_SIZE
            
            batch_files = []
            batch_counts = []
            
            # Process tiles in batches
            for batch_idx in range(n_batches):
                batch_start = batch_idx * BATCH_SIZE
                batch_end = min(batch_start + BATCH_SIZE, len(tiles))
                batch_tiles = tiles[batch_start:batch_end]
                
                if verbose:
                    print(f"  Batch {batch_idx+1}/{n_batches} ({len(batch_tiles)} tiles)...", end='\r')
                
                # Group tiles by image
                tiles_by_image = {}
                for img_idx, z_start, y_start, x_start in batch_tiles:
                    if img_idx not in tiles_by_image:
                        tiles_by_image[img_idx] = []
                    tiles_by_image[img_idx].append((z_start, y_start, x_start))
                
                # Extract tiles from each image
                batch_image_list = []
                batch_coords_list = []
                
                for img_idx in sorted(tiles_by_image.keys()):
                    img_path = image_paths[img_idx]
                    coord_path = coord_paths[img_idx]
                    
                    try:
                        # Open image with memory mapping
                        mmap_array = None
                        try:
                            mmap_array = memmap(img_path)
                            with TiffFile(img_path) as tif:
                                image_shape = tif.series[0].shape
                        except Exception:
                            continue
                        
                        # Load coordinates
                        coords = np.load(coord_path).astype(np.float32)
                        coords = remove_duplicate_coords_3d(coords)
                        
                        # Extract tiles
                        for z_start, y_start, x_start in tiles_by_image[img_idx]:
                            z_end = min(z_start + tile_size[0], image_shape[0])
                            y_end = min(y_start + tile_size[1], image_shape[1])
                            x_end = min(x_start + tile_size[2], image_shape[2])
                            
                            # Read tile
                            tile = np.ascontiguousarray(
                                mmap_array[z_start:z_end, y_start:y_end, x_start:x_end]
                            ).astype(np.float32)
                            
                            # Pad if needed
                            if tile.shape != tile_size:
                                pad_z = tile_size[0] - tile.shape[0]
                                pad_y = tile_size[1] - tile.shape[1]
                                pad_x = tile_size[2] - tile.shape[2]
                                tile = np.pad(tile, ((0, pad_z), (0, pad_y), (0, pad_x)), 
                                            mode='constant', constant_values=0)
                            
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
                            
                            batch_image_list.append(tile)
                            batch_coords_list.append(tile_coords)
                            
                            del tile, tile_coords
                        
                        del mmap_array, coords
                        gc.collect()
                        
                    except Exception as e:
                        if verbose:
                            print(f"\n    Warning: Failed to process image {img_idx}: {e}")
                        continue
                
                # Save batch
                batch_file = split_dir / f'batch_{batch_idx:04d}.npz'
                np.savez_compressed(batch_file, 
                                  images=batch_image_list, 
                                  coords=batch_coords_list)
                
                batch_files.append(str(batch_file.relative_to(output_dir)))
                batch_counts.append(len(batch_image_list))
                
                del batch_image_list, batch_coords_list
                gc.collect()
                
                if verbose:
                    print(f"  Batch {batch_idx+1}/{n_batches} saved ({batch_counts[-1]} tiles)")
            
            metadata['splits'][split_name] = {
                'batch_files': batch_files,
                'batch_counts': batch_counts,
                'total_tiles': sum(batch_counts)
            }
        finally:
            if step_context:
                step_context.__exit__(None, None, None)
    
    # Save metadata
    metadata_file = output_dir / 'metadata.json'
    with open(metadata_file, 'w') as f:
        # Convert numpy types to native Python types for JSON
        metadata_serializable = {
            'tile_size': list(tile_size),
            'total_tiles': int(total_tiles),
            'splits': {
                k: {
                    'batch_files': v['batch_files'],
                    'batch_counts': [int(c) for c in v['batch_counts']],
                    'total_tiles': int(v['total_tiles'])
                }
                for k, v in metadata['splits'].items()
            },
            'image_paths': image_paths,
            'coord_paths': coord_paths
        }
        json.dump(metadata_serializable, f, indent=2)
    
    if verbose:
        print(f"\n✓ Streaming dataset saved to {output_dir}")
        print(f"  Total tiles: {total_tiles}")
        for split_name, info in metadata['splits'].items():
            print(f"  {split_name}: {info['total_tiles']} tiles in {len(info['batch_files'])} batches")
    
    return str(output_dir)


def load_dataset_streaming(
    dataset_dir: str,
    split: str = 'train',
    batch_idx: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a batch from the streaming dataset.
    
    Parameters
    ----------
    dataset_dir : str
        Path to dataset directory
    split : str
        Split to load ('train', 'test', 'valid')
    batch_idx : Optional[int]
        Batch index to load. If None, returns all batches concatenated (may be memory intensive)
    
    Returns
    -------
    images : np.ndarray
        Array of images (object array)
    coords : np.ndarray
        Array of coordinates (object array)
    """
    dataset_dir = Path(dataset_dir)
    metadata_file = dataset_dir / 'metadata.json'
    
    if not metadata_file.exists():
        raise ValueError(f"Metadata file not found: {metadata_file}")
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    if split not in metadata['splits']:
        raise ValueError(f"Split '{split}' not found in dataset")
    
    split_info = metadata['splits'][split]
    batch_files = split_info['batch_files']
    
    if batch_idx is not None:
        # Load specific batch
        if batch_idx >= len(batch_files):
            raise ValueError(f"Batch index {batch_idx} out of range (max: {len(batch_files)-1})")
        
        batch_path = dataset_dir / batch_files[batch_idx]
        data = np.load(batch_path, allow_pickle=True)
        return data['images'], data['coords']
    else:
        # Load all batches (may be memory intensive)
        all_images = []
        all_coords = []
        
        for batch_file in batch_files:
            batch_path = dataset_dir / batch_file
            data = np.load(batch_path, allow_pickle=True)
            all_images.append(data['images'])
            all_coords.append(data['coords'])
        
        return np.concatenate(all_images), np.concatenate(all_coords)


def dataset_generator(
    dataset_dir: str,
    split: str = 'train',
    shuffle_batches: bool = True,
    rng_key: Optional[jax.Array] = None
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Generator that yields batches from the streaming dataset.
    Memory efficient - only loads one batch at a time.
    """
    dataset_dir = Path(dataset_dir)
    metadata_file = dataset_dir / 'metadata.json'
    
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    if split not in metadata['splits']:
        raise ValueError(f"Split '{split}' not found in dataset")
    
    batch_files = metadata['splits'][split]['batch_files']
    
    if shuffle_batches and rng_key is not None:
        perms = np.asarray(random.permutation(rng_key, len(batch_files)))
        batch_files = [batch_files[i] for i in perms]
    
    for batch_file in batch_files:
        batch_path = dataset_dir / batch_file
        data = np.load(batch_path, allow_pickle=True)
        yield data['images'], data['coords']