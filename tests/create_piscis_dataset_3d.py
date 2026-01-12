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
import gc

# Add Piscis3D to path
piscis3d_path = Path(__file__).parent.parent / "Piscis3D"
if str(piscis3d_path) not in sys.path:
    sys.path.insert(0, str(piscis3d_path))

try:
    print("DEBUG: About to import jax", flush=True)
    sys.stdout.flush()
    import jax
    print("DEBUG: jax imported successfully", flush=True)
    sys.stdout.flush()
    
    print("DEBUG: About to import jax.numpy", flush=True)
    sys.stdout.flush()
    import jax.numpy as jnp
    print("DEBUG: jax.numpy imported successfully", flush=True)
    sys.stdout.flush()
    
    print("DEBUG: About to import piscis3d modules", flush=True)
    sys.stdout.flush()
    from piscis3d.data import generate_dataset_3d_from_paths
    from piscis3d.data_streaming import generate_dataset_3d_streaming
    print("DEBUG: piscis3d modules imported successfully", flush=True)
    sys.stdout.flush()
except ImportError as e:
    print(f"Error importing Piscis3D modules: {e}", flush=True)
    print("Make sure Piscis3D is properly set up.", flush=True)
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
    tile_size: Tuple[int, int, int] = (8, 64, 64),
    min_spots: int = 1,
    train_size: float = 0.7,
    test_size: float = 0.15,
    random_seed: int = 42,
    exclude_conditions: List[str] = None,
    overlap_factor: float = 0.0,
    batch_size: int = 1,
    verbose: bool = True
):
    # Debug: Print immediately to verify function was called
    print("DEBUG: generate_piscis_dataset_3d() function started", flush=True)
    sys.stdout.flush()
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
    print("DEBUG: About to set JAX_PLATFORMS", flush=True)
    sys.stdout.flush()
    
    # Set JAX to use CPU if not already set (for dataset generation, CPU is sufficient)
    if 'JAX_PLATFORMS' not in os.environ:
        os.environ['JAX_PLATFORMS'] = 'cpu'
    
    print("DEBUG: JAX_PLATFORMS set, about to load dataset paths", flush=True)
    sys.stdout.flush()
    
    # Load only file paths - don't load images into memory
    if verbose:
        print("Loading dataset paths only (not loading images to save memory)...", flush=True)
    sys.stdout.flush()
    
    print("DEBUG: About to call load_dataset_paths_only()", flush=True)
    sys.stdout.flush()
    
    print("DEBUG: About to call load_dataset_paths_only() - BEFORE call", flush=True)
    sys.stdout.flush()
    
    try:
        dataset = load_dataset_paths_only(base_dir=base_dir, wavelength=wavelength, verbose=verbose)
        print("DEBUG: load_dataset_paths_only() returned - AFTER assignment", flush=True)
        sys.stdout.flush()
        print(f"DEBUG: Dataset type: {type(dataset)}, size: {len(dataset) if hasattr(dataset, '__len__') else 'N/A'}", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"DEBUG: ERROR in load_dataset_paths_only(): {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        raise
    
    print("DEBUG: About to force garbage collection", flush=True)
    sys.stdout.flush()
    
    # Force garbage collection after loading paths
    import gc
    gc.collect()
    print("DEBUG: Garbage collection completed", flush=True)
    sys.stdout.flush()
    
    print("DEBUG: About to collect image/coordinate paths", flush=True)
    sys.stdout.flush()
    
    # Collect image/coordinate paths from non-excluded conditions
    image_paths = []
    coord_paths = []
    
    print(f"DEBUG: Dataset has {len(dataset)} conditions", flush=True)
    sys.stdout.flush()
    
    for condition, data in dataset.items():
        print(f"DEBUG: Processing condition: {condition}", flush=True)
        sys.stdout.flush()
        if condition in (exclude_conditions or []):
            if verbose:
                print(f"Skipping {condition} (excluded)", flush=True)
            continue
        
        # Get paths from dataset
        img_paths = data.get('image_paths', [])
        spot_paths = data.get('spots_paths', [])
        
        print(f"DEBUG: Condition {condition} has {len(img_paths)} image paths", flush=True)
        sys.stdout.flush()
        
        # Both lists should be the same length
        for img_path, spot_path in zip(img_paths, spot_paths):
            image_paths.append(img_path)
            coord_paths.append(spot_path)
        
        if verbose:
            print(f"  {condition}: Added {len(img_paths)} image/spot pairs", flush=True)
    
    print(f"DEBUG: Finished collecting paths. Total: {len(image_paths)} images", flush=True)
    sys.stdout.flush()
    
    # Force garbage collection after collecting paths
    gc.collect()
    print("DEBUG: Garbage collection after path collection", flush=True)
    sys.stdout.flush()
    
    if len(image_paths) == 0:
        raise ValueError("No images found after filtering. Please check your data paths and exclusion criteria.")
    
    if len(image_paths) != len(coord_paths):
        raise ValueError(f"Mismatch: {len(image_paths)} images but {len(coord_paths)} coordinate files")
    
    print("DEBUG: Path validation passed", flush=True)
    sys.stdout.flush()
    
    if verbose:
        print(f"Found {len(image_paths)} images to process")
        print("Processing images incrementally from disk to reduce memory usage...")
    
    print("DEBUG: About to create output directory", flush=True)
    sys.stdout.flush()
    
    # Create output directory
    output_dir = Path(output_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    
    print("DEBUG: Output directory created", flush=True)
    sys.stdout.flush()
    
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
    
    # Initialize step-by-step memory profiler
    profiler = None
    profiler_log = None
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        from step_memory_profiler import initialize_profiler
        # Save log in same directory as output (or current directory if output_path is just a name)
        output_path_obj = Path(output_path)
        if output_path_obj.is_absolute():
            profiler_log = str(output_path_obj.parent / "memory_profiling.log")
        else:
            profiler_log = str(Path.cwd() / "memory_profiling.log")
        profiler = initialize_profiler(log_file=profiler_log, verbose=verbose)
        if verbose:
            print(f"\n{'='*70}")
            print(f"Step-by-step memory profiling enabled")
            print(f"Log file: {profiler_log}")
            print(f"{'='*70}\n")
    except ImportError as e:
        if verbose:
            print(f"Warning: Step-by-step memory profiler not available: {e}")
            print("(step_memory_profiler.py may not be in path)")
    except Exception as e:
        if verbose:
            print(f"Warning: Could not initialize memory profiler: {e}")
            import traceback
            traceback.print_exc()
    
    # Memory check and troubleshooting
    if verbose:
        print("\n" + "="*60)
        print("MEMORY CHECK AND TROUBLESHOOTING")
        print("="*60)
        
        # Import memory monitor
        try:
            from memory_monitor import (
                print_memory_status, 
                check_memory_before_processing,
                diagnose_memory_issue,
                estimate_tile_memory
            )
            
            # Print initial memory status
            print_memory_status("Initial")
            
            # Estimate memory needs
            estimated_tiles_per_image = 100  # Conservative estimate
            can_proceed, mem_message = check_memory_before_processing(
                tile_size=tile_size,
                n_images=len(image_paths),
                estimated_tiles_per_image=estimated_tiles_per_image,
                safety_margin=0.3  # 30% safety margin
            )
            
            print(mem_message)
            
            if not can_proceed:
                print("\n⚠️  WARNING: Insufficient memory detected!")
                print("Recommendations:")
                print(f"  1. Reduce tile size from {tile_size} to (8, 64, 64) or smaller")
                print(f"  2. Reduce max_tiles_per_image (currently 100)")
                print(f"  3. Process fewer images at a time")
                print("\nAttempting to continue anyway...")
                print("(This may cause OOM - monitor memory usage carefully)")
            
            # Show tile size impact
            tile_est = estimate_tile_memory(tile_size, len(image_paths) * estimated_tiles_per_image)
            print(f"\nTile Memory Analysis:")
            print(f"  Tile size: {tile_size}")
            print(f"  Bytes per tile: {tile_est['bytes_per_tile']:,} ({tile_est['bytes_per_tile'] / (1024*1024):.2f} MB)")
            print(f"  Estimated total tiles: {tile_est['n_tiles']:,}")
            print(f"  Estimated total tile data: {tile_est['total_bytes'] / (1024**3):.2f} GB")
            
            if tile_size[0] * tile_size[1] * tile_size[2] > 8 * 64 * 64:
                print(f"\n⚠️  WARNING: Tile size {tile_size} is LARGE!")
                print(f"  Recommended: (8, 64, 64) = {8*64*64:,} voxels")
                print(f"  Your size: {tile_size[0]*tile_size[1]*tile_size[2]:,} voxels")
                print(f"  Ratio: {tile_size[0]*tile_size[1]*tile_size[2] / (8*64*64):.1f}x larger")
            
        except ImportError:
            print("  (Memory monitor not available - skipping detailed checks)")
        except Exception as e:
            print(f"  (Memory check failed: {e})")
        
        print("="*60 + "\n")
    
    # Generate JAX random key
    print("DEBUG: About to check JAX devices", flush=True)
    sys.stdout.flush()
    
    if verbose:
        try:
            devices = jax.devices()
            print(f"JAX devices available: {[str(d) for d in devices]}")
        except Exception as e:
            print(f"Warning: Could not check JAX devices: {e}")
            print("Continuing with CPU mode...")
    
    print("DEBUG: About to generate JAX random key", flush=True)
    sys.stdout.flush()
    
    key = jax.random.PRNGKey(random_seed)
    
    print("DEBUG: JAX random key generated", flush=True)
    sys.stdout.flush()
    
    print("DEBUG: About to force garbage collection", flush=True)
    sys.stdout.flush()
    
    # Force garbage collection before starting
    gc.collect()
    
    print("DEBUG: Garbage collection completed", flush=True)
    sys.stdout.flush()
    
    # Choose dataset generation method based on expected size
    use_streaming = True  # Use streaming by default for large datasets
    
    print("DEBUG: About to calculate max_tiles_per_image", flush=True)
    sys.stdout.flush()
    
    # Adjust max_tiles_per_image based on tile size to prevent OOM
    # For 2GB images, be VERY aggressive with limits
    tile_voxels = tile_size[0] * tile_size[1] * tile_size[2]
    safe_voxels = 8 * 64 * 64
    
    # Ultra-conservative: Even with correct tile size, limit to 50 tiles per image
    # This prevents memory spikes when processing large images
    if tile_voxels <= safe_voxels:
        # Tile size is correct, but still limit aggressively for 2GB images
        max_tiles_per_image = 50  # Reduced from 100 - very conservative
        if verbose:
            print(f"✓ Tile size {tile_size} is correct (safe size)")
            print(f"  Using conservative limit: {max_tiles_per_image} tiles per image")
    else:
        # Larger tiles = even fewer tiles
        size_ratio = tile_voxels / safe_voxels
        max_tiles_per_image = max(10, int(50 / size_ratio))  # At least 10 tiles
        if verbose:
            print(f"⚠️  Large tile size detected: {tile_size}")
            print(f"   Reducing max_tiles_per_image to {max_tiles_per_image} to prevent OOM")
    
    print(f"DEBUG: About to start dataset generation. use_streaming={use_streaming}, max_tiles_per_image={max_tiles_per_image}", flush=True)
    sys.stdout.flush()
    
    if use_streaming:
        if verbose:
            print("Using STREAMING dataset generation (memory efficient)...")
            print(f"  Max tiles per image: {max_tiles_per_image}")
        
        print("DEBUG: About to call generate_dataset_3d_streaming()", flush=True)
        sys.stdout.flush()
        
        try:
            # Use streaming approach - saves batches without loading everything
            if profiler:
                with profiler.step("Generate Streaming Dataset", "generate_dataset_3d_streaming"):
                    generate_dataset_3d_streaming(
                        output_dir=str(output_path),
                        image_paths=image_paths,
                        coord_paths=coord_paths,
                        key=key,
                        tile_size=tile_size,
                        min_spots=min_spots,
                        train_size=train_size,
                        test_size=test_size,
                        overlap_factor=overlap_factor,
                        max_tiles_per_image=max_tiles_per_image,
                        verbose=verbose
                    )
            else:
                generate_dataset_3d_streaming(
                    output_dir=str(output_path),
                    image_paths=image_paths,
                    coord_paths=coord_paths,
                    key=key,
                    tile_size=tile_size,
                    min_spots=min_spots,
                    train_size=train_size,
                    test_size=test_size,
                    overlap_factor=overlap_factor,
                    max_tiles_per_image=max_tiles_per_image,
                    verbose=verbose
                )
            if verbose:
                print(f"\n✓ 3D Streaming Dataset successfully generated at: {output_path}")
                print(f"  This format can be loaded incrementally during training")
            
            # Stop profiler if it was started
            if profiler:
                try:
                    profiler.stop()
                    if verbose and profiler_log:
                        print(f"\n{'='*70}")
                        print(f"Memory profiling log saved to: {profiler_log}")
                        print(f"{'='*70}\n")
                except Exception as e:
                    if verbose:
                        print(f"Warning: Could not stop profiler: {e}")
        except MemoryError as e:
            # Stop profiler to save final state
            if profiler:
                try:
                    profiler.stop()
                    if verbose and profiler_log:
                        print(f"\n{'='*70}")
                        print(f"Memory profiling log saved to: {profiler_log}")
                        print(f"{'='*70}\n")
                except:
                    pass
            print(f"\n✗ OUT OF MEMORY ERROR!")
            print(f"  Error: {e}")
            print(f"\nTroubleshooting:")
            print(f"  1. Tile size used: {tile_size}")
            print(f"  2. Max tiles per image: {max_tiles_per_image}")
            print(f"  3. Total images: {len(image_paths)}")
            print(f"\nSolutions:")
            print(f"  - Reduce tile size to (8, 64, 64) or smaller")
            print(f"  - Reduce max_tiles_per_image further")
            print(f"  - Process fewer images at a time")
            print(f"  - Check available memory: free -h")
            
            # Try to diagnose
            try:
                from memory_monitor import diagnose_memory_issue
                diagnose_memory_issue()
            except:
                pass
            
            import traceback
            traceback.print_exc()
            raise
        except Exception as e:
            print(f"\n✗ Error generating streaming 3D dataset: {e}")
            print(f"\nTroubleshooting info:")
            print(f"  Tile size: {tile_size}")
            print(f"  Max tiles per image: {max_tiles_per_image}")
            print(f"  Total images: {len(image_paths)}")
            
            # Check if it might be memory-related
            if "memory" in str(e).lower() or "oom" in str(e).lower():
                print(f"\n⚠️  This looks like a memory issue!")
                print(f"  Try reducing tile size to (8, 64, 64)")
            
            import traceback
            traceback.print_exc()
            raise
    else:
        if verbose:
            print("Using STANDARD dataset generation...")
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
    # Debug: Print immediately to verify script is running
    print("DEBUG: main() function started", flush=True)
    sys.stdout.flush()
    
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
        default=[8, 64, 64],
        metavar=("DEPTH", "HEIGHT", "WIDTH"),
        help="Tile size for splitting images (z y x). Default: (8, 64, 64) for minimal memory. Original was (32, 256, 256) - this is 64x smaller"
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
    
    # Validate tile size - warn if too large
    tile_voxels = tile_size[0] * tile_size[1] * tile_size[2]
    safe_voxels = 8 * 64 * 64  # (8, 64, 64)
    
    if tile_voxels > safe_voxels:
        print(f"\n{'='*60}")
        print("⚠️  WARNING: TILE SIZE IS TOO LARGE!")
        print(f"{'='*60}")
        print(f"  Current tile size: {tile_size} = {tile_voxels:,} voxels")
        print(f"  Recommended: (8, 64, 64) = {safe_voxels:,} voxels")
        print(f"  Your tiles are {tile_voxels / safe_voxels:.1f}x larger!")
        print(f"  This will likely cause OUT OF MEMORY errors!")
        print(f"\n  To fix, run with:")
        print(f"    --tile_size 8 64 64")
        print(f"{'='*60}\n")
        print("ERROR: Tile size is too large. This will cause OOM errors.")
        print("Aborting. Please use smaller tile size (--tile_size 8 64 64).")
        sys.exit(1)
    
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
    # Debug: Print immediately to verify script entry point
    print("DEBUG: Script entry point reached", flush=True)
    sys.stdout.flush()
    
    try:
        main()
    except Exception as e:
        print(f"DEBUG: Exception in main(): {e}", file=sys.stderr, flush=True)
        import traceback
        traceback.print_exc()
        raise
