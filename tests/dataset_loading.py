"""
Dataset loading script for experiment data structure.

This script loads all images and spots from the following structure:
/scratch/qgs8612/experiment/
├── {condition}/          # e.g., 0hr_Amputation, 12hr_Incision
│   ├── Image1/
│   │   ├── 565/
│   │   │   ├── *.tif
│   │   │   └── results/
│   │   │       └── spots_post_decomposition_and_background_removed.npy
│   ├── Image2/...
│   └── Image3/...
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from tifffile import imread


def load_dataset(base_dir: str = "/scratch/qgs8612/experiment",
                 wavelength: str = "565",
                 verbose: bool = True) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Load all images and spots from the experiment directory structure.
    
    Parameters
    ----------
    base_dir : str
        Base directory path (default: "/scratch/qgs8612/experiment")
    wavelength : str
        Wavelength folder to load from (default: "565")
    verbose : bool
        Whether to print progress information
    
    Returns
    -------
    Dict[str, Dict[str, np.ndarray]]
        Dictionary with keys like '0hr_Amputation', '0hr_Incision', etc.
        Each value is a dict with keys:
            - 'images': List of numpy arrays (image data)
            - 'spots': List of numpy arrays (spots data)
            - 'image_paths': List of file paths for images
            - 'spots_paths': List of file paths for spots
    """
    base_path = Path(base_dir)
    
    if not base_path.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    # Subfolders (conditions)
    conditions = [
        "0hr_Amputation",
        "0hr_Incision",
        "6hr_Amputation",
        "6hr_Incision",
        "12hr_Amputation",
        "12hr_Incision"
    ]
    
    # Images
    images = ["Image1", "Image2", "Image3"]
    
    dataset = {}
    
    for condition in conditions:
        condition_path = base_path / condition
        
        if not condition_path.exists():
            if verbose:
                print(f"Warning: Condition folder not found: {condition_path}")
            continue
        
        # Initialize storage for this condition
        condition_images = []
        condition_spots = []
        image_paths = []
        spots_paths = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"Loading condition: {condition}")
            print(f"{'='*60}")
        
        for image_name in images:
            image_path = condition_path / image_name / wavelength
            
            if not image_path.exists():
                if verbose:
                    print(f"  Warning: Image folder not found: {image_path}")
                continue
            
            # Find .tif file in the wavelength folder
            tif_files = list(image_path.glob("*.tif"))
            if len(tif_files) == 0:
                if verbose:
                    print(f"  Warning: No .tif file found in {image_path}")
                continue
            elif len(tif_files) > 1:
                if verbose:
                    print(f"  Warning: Multiple .tif files found in {image_path}, using first one")
            
            tif_file = tif_files[0]
            
            # Find corresponding spots file
            results_dir = image_path / "results"
            spots_file = results_dir / "spots_post_decomposition_and_background_removed.npy"
            
            if not spots_file.exists():
                if verbose:
                    print(f"  Warning: Spots file not found: {spots_file}")
                # Still load the image even if spots are missing
                try:
                    img = imread(str(tif_file))
                    condition_images.append(img)
                    condition_spots.append(None)  # None for missing spots
                    image_paths.append(str(tif_file))
                    spots_paths.append(None)
                    if verbose:
                        print(f"  ✓ Loaded: {image_name} (image only, no spots)")
                except Exception as e:
                    if verbose:
                        print(f"  ✗ Error loading {image_name}: {e}")
                continue
            
            # Load both image and spots
            try:
                if verbose:
                    print(f"  Loading {image_name}...")
                
                # Load image
                img = imread(str(tif_file))
                if verbose:
                    print(f"    Image shape: {img.shape}")
                
                # Load spots
                spots = np.load(str(spots_file))
                if verbose:
                    print(f"    Spots shape: {spots.shape if spots.ndim > 0 else 'scalar'}")
                    if spots.ndim > 0:
                        print(f"    Number of spots: {len(spots)}")
                
                condition_images.append(img)
                condition_spots.append(spots)
                image_paths.append(str(tif_file))
                spots_paths.append(str(spots_file))
                
                if verbose:
                    print(f"    ✓ Successfully loaded {image_name}")
                    
            except Exception as e:
                if verbose:
                    print(f"  ✗ Error loading {image_name}: {e}")
                continue
        
        # Store for this condition
        if len(condition_images) > 0:
            # Create variable names like "0hr_Amputation_images" and "0hr_Amputation_spots"
            dataset[condition] = {
                'images': condition_images,
                'spots': condition_spots,
                'image_paths': image_paths,
                'spots_paths': spots_paths
            }
            
            if verbose:
                print(f"\n  Summary for {condition}:")
                print(f"    Images loaded: {len(condition_images)}")
                print(f"    Spots loaded: {sum(1 for s in condition_spots if s is not None)}")
        else:
            if verbose:
                print(f"  No data loaded for {condition}")
    
    return dataset


def create_flat_arrays(dataset: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    """
    Create flat arrays from the dataset for easier access.
    
    This creates variables like:
    - 0hr_Amputation_images: List of all images for 0hr_Amputation
    - 0hr_Amputation_spots: List of all spots for 0hr_Amputation
    
    Parameters
    ----------
    dataset : Dict[str, Dict[str, np.ndarray]]
        Dataset dictionary from load_dataset()
    
    Returns
    -------
    Dict[str, np.ndarray or List]
        Dictionary with flat arrays/lists for each condition
    """
    flat_data = {}
    
    for condition, data in dataset.items():
        # Convert condition name to valid variable-like name
        # e.g., "0hr_Amputation" -> "0hr_Amputation" (already valid)
        var_name_base = condition
        
        flat_data[f"{var_name_base}_images"] = data['images']
        flat_data[f"{var_name_base}_spots"] = data['spots']
        flat_data[f"{var_name_base}_image_paths"] = data['image_paths']
        flat_data[f"{var_name_base}_spots_paths"] = data['spots_paths']
    
    return flat_data


def load_dataset_as_arrays(base_dir: str = "/scratch/qgs8612/experiment",
                           wavelength: str = "565",
                           verbose: bool = True,
                           return_dict: bool = False) -> Dict[str, List]:
    """
    Main function to load dataset and return as flat arrays.
    
    Parameters
    ----------
    base_dir : str
        Base directory path
    wavelength : str
        Wavelength folder to load from
    verbose : bool
        Whether to print progress information
    return_dict : bool
        If True, return nested dict structure; if False, return flat arrays
    
    Returns
    -------
    Dict[str, List]
        Dictionary with keys like '0hr_Amputation_images', '0hr_Amputation_spots', etc.
        Each value is a list of numpy arrays.
    """
    dataset = load_dataset(base_dir, wavelength, verbose)
    
    if return_dict:
        return dataset
    
    flat_data = create_flat_arrays(dataset)
    return flat_data


def main():
    """Example usage of the dataset loading functions."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Load experiment dataset")
    parser.add_argument("--base_dir", type=str, 
                       default="/scratch/qgs8612/experiment",
                       help="Base directory path")
    parser.add_argument("--wavelength", type=str, default="565",
                       help="Wavelength folder to load (default: 565)")
    parser.add_argument("--quiet", action="store_true",
                       help="Suppress verbose output")
    
    args = parser.parse_args()
    
    # Load dataset
    data = load_dataset_as_arrays(
        base_dir=args.base_dir,
        wavelength=args.wavelength,
        verbose=not args.quiet
    )
    
    # Print summary
    print("\n" + "="*60)
    print("Dataset Loading Summary")
    print("="*60)
    for key in sorted(data.keys()):
        if key.endswith('_images'):
            condition = key.replace('_images', '')
            n_images = len(data[key])
            n_spots = len([s for s in data[condition + '_spots'] if s is not None])
            print(f"{condition}:")
            print(f"  Images: {n_images}")
            print(f"  Spots:  {n_spots}")
    
    print("\n" + "="*60)
    print("Usage Example:")
    print("="*60)
    print("# After loading:")
    print("data = load_dataset_as_arrays()")
    print("# Access data like:")
    print("images = data['0hr_Amputation_images']")
    print("spots = data['0hr_Amputation_spots']")
    print("# images[0] is the first image for 0hr_Amputation")
    print("# spots[0] is the corresponding spots array")
    
    return data


if __name__ == "__main__":
    data = main()
