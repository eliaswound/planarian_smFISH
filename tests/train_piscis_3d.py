"""
Train 3D Piscis model for spot detection.

This script trains a 3D Piscis model using the streaming dataset format.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import numpy as np
import jax
import jax.numpy as jnp
from jax import random

# Add Piscis3D to path
sys.path.insert(0, str(Path(__file__).parent.parent / "Piscis3D"))

try:
    from piscis3d.data_streaming import load_dataset_streaming, dataset_generator
    from piscis3d.models.spots_3d import SpotsModel3D
    from piscis3d.transforms import voronoi_transform_3d, batch_adjust_3d
    HAS_PISCIS3D = True
except ImportError as e:
    HAS_PISCIS3D = False
    print(f"ERROR: Piscis3D modules not found: {e}")
    print("Make sure Piscis3D directory exists and contains the required modules.")


def create_training_directory(base_dir: str = "/scratch/qgs8612/piscis3d_dataset"):
    """
    Create training directory structure.
    
    Parameters
    ----------
    base_dir : str
        Base directory for Piscis3D training outputs
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for models and logs
    models_dir = base_path / "models"
    logs_dir = base_path / "logs"
    checkpoints_dir = base_path / "checkpoints"
    
    models_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)
    checkpoints_dir.mkdir(exist_ok=True)
    
    print(f"Created training directory structure at: {base_dir}")
    print(f"  Models: {models_dir}")
    print(f"  Logs: {logs_dir}")
    print(f"  Checkpoints: {checkpoints_dir}")
    
    return base_path


def main():
    """Command-line interface for 3D Piscis model training."""
    parser = argparse.ArgumentParser(
        description="Train a 3D Piscis spot detection model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to train"
    )
    
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the streaming dataset directory"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/qgs8612/piscis3d_dataset",
        help="Directory where model outputs will be saved"
    )
    
    parser.add_argument(
        "--tile_size",
        type=int,
        nargs=3,
        default=[8, 64, 64],
        metavar=("DEPTH", "HEIGHT", "WIDTH"),
        help="Size of input tiles (z y x)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size for training (smaller for 3D due to memory constraints)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="Learning rate for optimizer"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    if not HAS_PISCIS3D:
        print("ERROR: Piscis3D modules not available.")
        sys.exit(1)
    
    # Validate dataset path
    dataset_path_obj = Path(args.dataset_path)
    if not dataset_path_obj.exists():
        print(f"ERROR: Dataset path not found: {args.dataset_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir_obj = create_training_directory(args.output_dir)
    
    tile_size = tuple(args.tile_size)
    
    if not args.quiet:
        print(f"\n{'='*60}")
        print("3D Piscis Model Training Configuration")
        print(f"{'='*60}")
        print(f"Model name: {args.model_name}")
        print(f"Dataset path: {args.dataset_path}")
        print(f"Output directory: {args.output_dir}")
        print(f"\nModel Parameters:")
        print(f"  Tile size (z, y, x): {tile_size}")
        print(f"\nTraining Parameters:")
        print(f"  Epochs: {args.epochs}")
        print(f"  Batch size: {args.batch_size}")
        print(f"  Learning rate: {args.learning_rate}")
        print(f"  Random seed: {args.random_seed}")
        print(f"{'='*60}\n")
    
    # Initialize random key
    key = random.PRNGKey(args.random_seed)
    
    # Test dataset loading
    if not args.quiet:
        print("Testing dataset loading...")
    
    try:
        # Load a small batch to test
        train_images, train_coords = load_dataset_streaming(
            dataset_dir=str(args.dataset_path),
            split='train',
            batch_idx=0
        )
        
        if not args.quiet:
            print(f"✓ Successfully loaded training batch:")
            print(f"  Images: {len(train_images)} tiles")
            print(f"  Coordinates: {len(train_coords)} coordinate arrays")
            if len(train_images) > 0:
                print(f"  Sample image shape: {train_images[0].shape}")
                print(f"  Sample coords shape: {train_coords[0].shape}")

        # Load metadata for overall dataset stats
        metadata_file = Path(args.dataset_path) / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
            if not args.quiet:
                print(f"\nDataset metadata:")
                print(f"  Tile size: {tuple(metadata.get('tile_size', []))}")
                print(f"  Total tiles: {metadata.get('total_tiles', 'N/A')}")
                splits = metadata.get("splits", {})
                for split_name, info in splits.items():
                    print(
                        f"  {split_name}: {info.get('total_tiles', 'N/A')} "
                        f"tiles in {len(info.get('batch_files', []))} batches"
                    )
        else:
            print(f"WARNING: metadata.json not found in {args.dataset_path}")

        # Test dataset generator for a few batches
        if not args.quiet:
            print("\nTesting dataset_generator for 'train' split (first 2 batches)...")
        gen = dataset_generator(str(args.dataset_path), split="train", shuffle_batches=False, rng_key=None)
        for i, (imgs_batch, coords_batch) in enumerate(gen):
            if i >= 2:
                break
            if not args.quiet:
                print(f"  Generator batch {i+1}:")
                print(f"    Images: {len(imgs_batch)} tiles")
                if len(imgs_batch) > 0:
                    print(f"    Sample image shape: {imgs_batch[0].shape}")
                print(f"    Coords arrays: {len(coords_batch)}")
                if len(coords_batch) > 0:
                    print(f"    Sample coords shape: {coords_batch[0].shape}")

    except Exception as e:
        print(f"ERROR: Failed to load dataset: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("NOTE: Full training loop not yet implemented.")
    print("Dataset loading and streaming generator work correctly.")
    print("To implement training, you need to:")
    print("  1. Create a 3D training loop similar to piscis.training.train_model")
    print("  2. Implement loss functions for 3D")
    print("  3. Set up optimizer and training state")
    print("  4. Implement checkpoint saving")
    print(f"{'='*60}\n")
    
    print("For now, you can use the dataset with:")
    print(f"  from piscis3d.data_streaming import dataset_generator")
    print(f"  generator = dataset_generator('{args.dataset_path}', split='train')")
    print(f"  for images, coords in generator:")
    print(f"      # Process batch here")


if __name__ == "__main__":
    main()
