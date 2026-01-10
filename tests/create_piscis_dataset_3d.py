"""
Create 3D Piscis training dataset from experiment data.

This script:
1. Loads all 3D images and spots using dataset_loading.py
2. Excludes specified conditions (default: 0hr conditions)
3. Generates a 3D Piscis training dataset using custom 3D tiling
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np

# Add Piscis3D to path
piscis3d_path = Path(__file__).parent.parent / "Piscis3D"
if str(piscis3d_path) not in sys.path:
    sys.path.insert(0, str(piscis3d_path))

try:
    import jax
    import jax.numpy as jnp
    from piscis3d.data import generate_dataset_3d_from_paths
except ImportError as e:
    print(f"Error importing Piscis3D modules: {e}")
    print("Make sure Piscis3D is properly set up.")
    sys.exit(1)

# Import dataset loading functions
try:
    from dataset_loading import load_dataset_paths_only
except ImportError:
    import sys
    from pathlib import Path
    tests_dir = Path(__file__).parent
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from dataset_loading import load_dataset_paths_only


def prepare_data_for_piscis_3d(dataset: dict, exclude_conditions: List[str] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Prepare 3D images and coordinates from dataset for Piscis3D.
    
    Parameters
    ----------
    dataset : dict
        Dataset dictionary from load_dataset()
    exclude_conditions : List[str], optional
        List of condition names to exclude (default: ['0hr_Amputation', '0hr_Incision'])
    
    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]]
        (images, coords) tuple where:
        - images: List of 3D image arrays with shape (z, y, x)
        - coords: List of spot coordinate arrays with shape (n_spots, 3) where columns are (z, y, x)
    """
    if exclude_conditions is None:
        exclude_conditions = ['0hr_Amputation', '0hr_Incision']
    
    if exclude_conditions == []:
        exclude_conditions = []
    
    all_images = []
    all_coords = []
    
    print(f"\n{'='*60}")
    print("Preparing 3D data for Piscis dataset generation")
    print(f"{'='*60}")
    print(f"Excluding conditions: {exclude_conditions}")
    print()
    
    for condition, data in dataset.items():
        if condition in exclude_conditions:
            print(f"Skipping {condition} (excluded)")
            continue
        
        images = data['images']
        spots = data['spots']
        
        print(f"\nProcessing {condition}:")
        print(f"  Total images: {len(images)}")
        
        # Process each image-spot pair
        for i, (img, spot_data) in enumerate(zip(images, spots)):
            if spot_data is None:
                print(f"  Warning: Image {i+1} has no spots data, skipping")
                continue
            
            # Validate and convert spots format
            if isinstance(spot_data, np.ndarray):
                # Check if spots array is empty
                if spot_data.size == 0:
                    print(f"  Warning: Image {i+1} has empty spots array, skipping")
                    continue
                
                # Ensure spots is 2D array
                if spot_data.ndim == 1:
                    print(f"  Warning: Image {i+1} has 1D spots array, skipping")
                    continue
                
                # For 3D training, we keep images and coordinates in 3D format
                # Images are already in format (z, y, x)
                if img.ndim != 3:
                    print(f"  Warning: Image {i+1} has {img.ndim} dimensions, expected 3 (z, y, x), skipping")
                    continue
                
                # Coordinates should be in format (n_spots, 3) where columns are (z, y, x)
                if spot_data.shape[1] != 3:
                    print(f"  Warning: Image {i+1} coordinates have shape {spot_data.shape}, expected (n_spots, 3), skipping")
                    continue
                
                # Ensure coordinates are float32 for consistency
                coords_3d = spot_data.astype(np.float32)
                
                all_images.append(img)
                all_coords.append(coords_3d)
                print(f"  ✓ Added Image {i+1}: shape={img.shape}, spots={coords_3d.shape}")
            else:
                print(f"  Warning: Image {i+1} has invalid spots format, skipping")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total images: {len(all_images)}")
    print(f"  Total spot arrays: {len(all_coords)}")
    print(f"{'='*60}\n")
    
    return all_images, all_coords


def generate_piscis_dataset_3d(
    base_dir: str = "/scratch/qgs8612/Experiment",
    output_path: str = "/scratch/qgs8612/piscis_training_dataset_3d",
    wavelength: str = "565",
    tile_size: Tuple[int, int, int] = (16, 128, 128),
    min_spots: int = 1,
    train_size: float = 0.7,
    test_size: float = 0.15,
    random_seed: int = 42,
    exclude_conditions: List[str] = None,
    overlap_factor: float = 0.0,
    batch_size: int = 1,
    verbose: bool = True
):
    """
    Generate 3D Piscis training dataset from experiment data.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing experiment data
    output_path : str
        Path where the dataset will be saved
    wavelength : str
        Wavelength folder to load from (default: "565")
    tile_size : Tuple[int, int, int]
        Tile size for splitting images (z, y, x) (default: (32, 256, 256))
    min_spots : int
        Minimum number of spots per tile (default: 1)
    train_size : float
        Fraction of dataset for training (default: 0.7)
    test_size : float
        Fraction of dataset for testing (default: 0.15)
        Validation size will be 1 - train_size - test_size
    random_seed : int
        Random seed for dataset splitting (default: 42)
    exclude_conditions : List[str]
        Conditions to exclude from dataset (default: ['0hr_Amputation', '0hr_Incision'])
    verbose : bool
        Whether to print progress information
    """
    # Set JAX to use CPU if not already set (for dataset generation, CPU is sufficient)
    if 'JAX_PLATFORMS' not in os.environ:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    
    # Load only file paths - don't load images into memory
    if verbose:
        print("Loading dataset paths only (not loading images to save memory)...")
    dataset = load_dataset_paths_only(base_dir=base_dir, wavelength=wavelength, verbose=verbose)
    
    # Collect image/coordinate paths from non-excluded conditions
    image_paths = []
    coord_paths = []
    for condition, data in dataset.items():
        if condition in (exclude_conditions or []):
            if verbose:
                print(f"Skipping {condition} (excluded)")
            continue
        
        # Get paths from dataset
        img_paths = data.get('image_paths', [])
        spot_paths = data.get('spots_paths', [])
        
        # Both lists should be the same length
        for img_path, spot_path in zip(img_paths, spot_paths):
            image_paths.append(img_path)
            coord_paths.append(spot_path)
        
        if verbose:
            print(f"  {condition}: Added {len(img_paths)} image/spot pairs")
    
    if len(image_paths) == 0:
        raise ValueError("No images found after filtering. Please check your data paths and exclusion criteria.")
    
    if len(image_paths) != len(coord_paths):
        raise ValueError(f"Mismatch: {len(image_paths)} images but {len(coord_paths)} coordinate files")
    
    if verbose:
        print(f"Found {len(image_paths)} images to process")
        print("Processing images incrementally from disk to reduce memory usage...")
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"\n{'='*60}")
        print("Generating 3D Piscis dataset")
        print(f"{'='*60}")
        print(f"Output path: {output_path}")
        print(f"Total images: {len(image_paths)}")
        print(f"Tile size (z, y, x): {tile_size}")
        print(f"Min spots per tile: {min_spots}")
        print(f"Train size: {train_size}")
        print(f"Test size: {test_size}")
        print(f"Validation size: {1.0 - train_size - test_size:.2f}")
        print(f"Random seed: {random_seed}")
        print(f"Overlap factor: {overlap_factor}")
        print(f"Batch size: {batch_size}")
        print(f"{'='*60}\n")
    
    # Generate JAX random key
    if verbose:
        try:
            devices = jax.devices()
            print(f"JAX devices available: {[str(d) for d in devices]}")
        except Exception as e:
            print(f"Warning: Could not check JAX devices: {e}")
            print("Continuing with CPU mode...")
    
    key = jax.random.PRNGKey(random_seed)
    
    # Generate dataset using Piscis3D (memory-optimized version)
    if verbose:
        print("Calling piscis3d.data.generate_dataset_3d_from_paths()...")
    try:
        # Process images incrementally from disk with memory optimization
        generate_dataset_3d_from_paths(
            path=str(output_path),
            image_paths=image_paths,
            coord_paths=coord_paths,
            key=key,
            tile_size=tile_size,
            min_spots=min_spots,
            train_size=train_size,
            test_size=test_size,
            overlap_factor=overlap_factor,
            batch_size=batch_size,
            verbose=verbose
        )
        if verbose:
            print(f"\n✓ 3D Dataset successfully generated at: {output_path}")
    except Exception as e:
        print(f"\n✗ Error generating 3D dataset: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    """Command-line interface for 3D dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate 3D Piscis training dataset from experiment data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/scratch/qgs8612/Experiment",
        help="Base directory containing experiment data"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="/scratch/qgs8612/piscis_training_dataset_3d",
        help="Path where the dataset will be saved"
    )
    
    parser.add_argument(
        "--wavelength",
        type=str,
        default="565",
        help="Wavelength folder to load from"
    )
    
    parser.add_argument(
        "--tile_size",
        type=int,
        nargs=3,
        default=[16, 128, 128],
        metavar=("DEPTH", "HEIGHT", "WIDTH"),
        help="Tile size for splitting images (z y x). Default: (16, 128, 128) for memory efficiency. Original was (32, 256, 256)"
    )
    
    parser.add_argument(
        "--min_spots",
        type=int,
        default=1,
        help="Minimum number of spots per tile"
    )
    
    parser.add_argument(
        "--train_size",
        type=float,
        default=0.7,
        help="Fraction of dataset for training"
    )
    
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.15,
        help="Fraction of dataset for testing"
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for dataset splitting"
    )
    
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="+",
        default=["0hr_Amputation", "0hr_Incision"],
        help="Conditions to exclude from dataset"
    )
    
    parser.add_argument(
        "--overlap_factor",
        type=float,
        default=0.0,
        help="Tile overlap factor (0.0 = no overlap, 0.5 = 50%% overlap). Lower = fewer tiles = less memory. Default: 0.0"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of tiles to accumulate before writing. Default: 1 (writes each tile immediately for minimal memory)"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate train/test sizes
    if args.train_size + args.test_size >= 1.0:
        parser.error("train_size + test_size must be less than 1.0")
    
    # Convert tile_size to tuple
    tile_size = tuple(args.tile_size)
    
    # Generate dataset with memory optimization parameters
    generate_piscis_dataset_3d(
        base_dir=args.base_dir,
        output_path=args.output_path,
        wavelength=args.wavelength,
        tile_size=tile_size,
        min_spots=args.min_spots,
        train_size=args.train_size,
        test_size=args.test_size,
        random_seed=args.random_seed,
        exclude_conditions=args.exclude if args.exclude else [],
        overlap_factor=args.overlap_factor,
        batch_size=args.batch_size,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
