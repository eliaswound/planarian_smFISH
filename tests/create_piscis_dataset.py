"""
Create Piscis training dataset from experiment data.

This script:
1. Loads all images and spots using dataset_loading.py
2. Excludes 0hr conditions (0hr_Amputation, 0hr_Incision)
3. Combines all remaining data (6hr and 12hr conditions)
4. Generates a Piscis training dataset using piscis.data.generate_dataset()
"""

import os
# Set JAX to use CPU if not already set (for dataset generation, CPU is sufficient)
# This prevents CUDA library errors during dataset generation
# Must be set BEFORE importing jax
if 'JAX_PLATFORMS' not in os.environ:
    os.environ['JAX_PLATFORMS'] = 'cpu'

import argparse
from pathlib import Path
from typing import List, Tuple
import numpy as np
import jax
import jax.numpy as jnp

# Import dataset loading functions
# Note: Run this script from the tests directory or add tests to PYTHONPATH
try:
    from dataset_loading import load_dataset
except ImportError:
    import sys
    from pathlib import Path
    # Add tests directory to path
    tests_dir = Path(__file__).parent
    if str(tests_dir) not in sys.path:
        sys.path.insert(0, str(tests_dir))
    from dataset_loading import load_dataset


def prepare_data_for_piscis(dataset: dict, exclude_conditions: List[str] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Prepare images and coordinates from dataset for Piscis.
    
    Parameters
    ----------
    dataset : dict
        Dataset dictionary from load_dataset()
    exclude_conditions : List[str], optional
        List of condition names to exclude (default: ['0hr_Amputation', '0hr_Incision'])
        If empty list or None, no conditions will be excluded.
    
    Returns
    -------
    Tuple[List[np.ndarray], List[np.ndarray]]
        (images, coords) tuple where:
        - images: List of image arrays
        - coords: List of spot coordinate arrays
    """
    if exclude_conditions is None:
        exclude_conditions = ['0hr_Amputation', '0hr_Incision']
    
    # If exclude_conditions is an empty list, don't exclude anything
    if exclude_conditions == []:
        exclude_conditions = []
    
    all_images = []
    all_coords = []
    
    print(f"\n{'='*60}")
    print("Preparing data for Piscis dataset generation")
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
            # Piscis expects coordinates in format (N, 3) or (N, 2) for 3D/2D
            if isinstance(spot_data, np.ndarray):
                # Check if spots array is empty
                if spot_data.size == 0:
                    print(f"  Warning: Image {i+1} has empty spots array, skipping")
                    continue
                
                # Ensure spots is 2D array
                if spot_data.ndim == 1:
                    # If 1D, might be a single coordinate - skip for now or reshape
                    print(f"  Warning: Image {i+1} has 1D spots array, skipping")
                    continue
                
                # Piscis expects (y, x) or (z, y, x) format
                # If spots are in different format, may need transformation
                all_images.append(img)
                all_coords.append(spot_data)
                print(f"  ✓ Added Image {i+1}: shape={img.shape}, spots={spot_data.shape}")
            else:
                print(f"  Warning: Image {i+1} has invalid spots format, skipping")
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total images: {len(all_images)}")
    print(f"  Total spot arrays: {len(all_coords)}")
    print(f"{'='*60}\n")
    
    return all_images, all_coords


def generate_piscis_dataset(
    base_dir: str = "/scratch/qgs8612/experiment",
    output_path: str = "/scratch/qgs8612/piscis_training_dataset",
    wavelength: str = "565",
    tile_size: Tuple[int, int] = (256, 256),
    min_spots: int = 1,
    train_size: float = 0.7,
    test_size: float = 0.15,
    random_seed: int = 42,
    exclude_conditions: List[str] = None,
    verbose: bool = True
):
    """
    Generate Piscis training dataset from experiment data.
    
    Parameters
    ----------
    base_dir : str
        Base directory containing experiment data
    output_path : str
        Path where the dataset will be saved
    wavelength : str
        Wavelength folder to load from (default: "565")
    tile_size : Tuple[int, int]
        Tile size for splitting images (default: (256, 256))
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
    try:
        import piscis
        from piscis import data as piscis_data
    except ImportError:
        raise ImportError("Piscis is not installed. Please install it: pip install git+https://github.com/zjniu/Piscis.git")
    
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(base_dir=base_dir, wavelength=wavelength, verbose=verbose)
    
    # Prepare images and coordinates
    images, coords = prepare_data_for_piscis(dataset, exclude_conditions=exclude_conditions)
    
    if len(images) == 0:
        raise ValueError("No images found after filtering. Please check your data paths and exclusion criteria.")
    
    if len(images) != len(coords):
        raise ValueError(f"Mismatch between number of images ({len(images)}) and coordinates ({len(coords)})")
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("Generating Piscis dataset")
    print(f"{'='*60}")
    print(f"Output path: {output_path}")
    print(f"Total images: {len(images)}")
    print(f"Tile size: {tile_size}")
    print(f"Min spots per tile: {min_spots}")
    print(f"Train size: {train_size}")
    print(f"Test size: {test_size}")
    print(f"Validation size: {1.0 - train_size - test_size:.2f}")
    print(f"Random seed: {random_seed}")
    print(f"{'='*60}\n")
    
    # Generate JAX random key
    # JAX_PLATFORMS should be set to 'cpu' by the bash script or at module import
    # This prevents CUDA library initialization errors during dataset generation
    if verbose:
        try:
            devices = jax.devices()
            print(f"JAX devices available: {[str(d) for d in devices]}")
        except Exception as e:
            print(f"Warning: Could not check JAX devices: {e}")
            print("Continuing with CPU mode...")
    
    key = jax.random.PRNGKey(random_seed)
    
    # Generate dataset using Piscis
    print("Calling piscis.data.generate_dataset()...")
    try:
        piscis_data.generate_dataset(
            path=str(output_path),
            images=images,
            coords=coords,
            key=key,
            tile_size=tile_size,
            min_spots=min_spots,
            train_size=train_size,
            test_size=test_size
        )
        print(f"\n✓ Dataset successfully generated at: {output_path}")
    except Exception as e:
        print(f"\n✗ Error generating dataset: {e}")
        raise


def main():
    """Command-line interface for dataset generation."""
    parser = argparse.ArgumentParser(
        description="Generate Piscis training dataset from experiment data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--base_dir",
        type=str,
        default="/scratch/qgs8612/experiment",
        help="Base directory containing experiment data"
    )
    
    parser.add_argument(
        "--output_path",
        type=str,
        default="/scratch/qgs8612/piscis_training_dataset",
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
        nargs=2,
        default=[256, 256],
        metavar=("HEIGHT", "WIDTH"),
        help="Tile size for splitting images (height width)"
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
    
    # Generate dataset
    generate_piscis_dataset(
        base_dir=args.base_dir,
        output_path=args.output_path,
        wavelength=args.wavelength,
        tile_size=tile_size,
        min_spots=args.min_spots,
        train_size=args.train_size,
        test_size=args.test_size,
        random_seed=args.random_seed,
        exclude_conditions=args.exclude,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
