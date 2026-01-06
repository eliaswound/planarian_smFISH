#!/usr/bin/env python3
"""
Test Piscis deep learning spot detection on smFISH images.

Piscis is an automatic deep learning algorithm for spot detection in 
fluorescence microscopy images. This script uses the built-in pre-trained
model from Hugging Face to detect spots.
"""

import numpy as np
from tifffile import imread, imwrite
from pathlib import Path
import yaml
import sys
import json
import time

try:
    import piscis
    HAS_PISCIS = True
except ImportError:
    HAS_PISCIS = False
    print("ERROR: Piscis is not installed!")
    print("Please install it with: pip install piscis")
    sys.exit(1)


def generate_coordinates_2D(y, x, shape, iteration=4, get_inner_spot=False):
    """Generate coordinates for 2D spot plotting."""
    coordinates_collection = [(y, x)]
    max_y = shape[0] - 1
    max_x = shape[1] - 1

    for _ in range(iteration):
        current_coordinates = coordinates_collection.copy()
        for coord in current_coordinates:
            cy, cx = coord
            if cx + 1 <= max_x:
                coordinates_collection.append((cy, cx + 1))
            if 0 <= cx - 1:
                coordinates_collection.append((cy, cx - 1))
            if cy + 1 <= max_y:
                coordinates_collection.append((cy + 1, cx))
            if 0 <= cy - 1:
                coordinates_collection.append((cy - 1, cx))
    
    coordinates_collection = list(set(coordinates_collection))
    
    if get_inner_spot:
        return coordinates_collection
    else:
        coordinates_collection = [
            coord for coord in coordinates_collection 
            if abs(coord[0] - y) + abs(coord[1] - x) == iteration
        ]
        return coordinates_collection


def create_spot_plot(image, spots, plot_spot_size=4, plot_inner_circle=False, plot_spot_label=False):
    """Create a 3D spot plot from detected spots."""
    z, y, x = image.shape
    
    if plot_spot_label:
        spot_plot = np.zeros(image.shape, dtype=np.uint32)
    else:
        spot_plot = np.zeros(image.shape, dtype=np.uint8)
    
    shape_2d = [spot_plot.shape[1], spot_plot.shape[2]]
    
    for i, spot in enumerate(spots):
        z_coord = int(spot[0])
        y_coord = int(spot[1])
        x_coord = int(spot[2])
        
        plot_locations = generate_coordinates_2D(
            y_coord, x_coord, 
            shape_2d, 
            iteration=plot_spot_size,
            get_inner_spot=plot_inner_circle
        )
        
        for plot_y, plot_x in plot_locations:
            if 0 <= z_coord < z and 0 <= plot_y < y and 0 <= plot_x < x:
                if plot_spot_label:
                    spot_plot[z_coord, plot_y, plot_x] = i + 1
                else:
                    spot_plot[z_coord, plot_y, plot_x] = 255
    
    return spot_plot


def predict_spots_piscis(image_path, output_dir, config):
    """
    Use Piscis to predict spots in an image.
    
    Args:
        image_path: Path to input image
        output_dir: Directory to save results
        config: Configuration dictionary
    """
    print("="*60)
    print("Piscis Spot Detection")
    print("="*60)
    print(f"Image: {image_path}")
    print(f"Output: {output_dir}")
    print("="*60)
    
    # Load image
    print("\nLoading image...")
    start_load = time.time()
    image = imread(image_path)
    load_time = time.time() - start_load
    print(f"Image shape: {image.shape} [Z, Y, X]")
    print(f"Load time: {load_time:.2f}s")
    
    # Get Piscis parameters from config
    model_name = config.get('model_name', None)  # None = use default built-in model
    batch_size = config.get('batch_size', 1)
    device = config.get('device', 'cpu')  # 'cpu', 'gpu', or 'tpu'
    
    print(f"\nPiscis parameters:")
    print(f"  Model: {model_name if model_name else 'Default built-in model'}")
    print(f"  Batch size: {batch_size}")
    print(f"  Device: {device}")
    
    # Run Piscis prediction
    print(f"\nRunning Piscis spot detection...")
    start_pred = time.time()
    
    try:
        # Piscis API: Use command-line interface via subprocess
        # This is the most reliable way as the Python API structure may vary
        import subprocess
        import tempfile
        
        # Save image to temp file
        with tempfile.NamedTemporaryFile(suffix='.tif', delete=False) as tmp_img:
            imwrite(tmp_img.name, image)
            tmp_img_path = tmp_img.name
        
        # Create temp output directory for Piscis
        tmp_output_dir = tempfile.mkdtemp()
        tmp_out_path = Path(tmp_output_dir) / "predictions.npy"
        
        # Run piscis predict command
        # Piscis CLI: piscis predict INPUT_PATH OUTPUT_PATH [OPTIONS]
        cmd = [
            'piscis', 'predict',
            tmp_img_path,
            str(tmp_out_path)
        ]
        
        # Add optional parameters
        if model_name:
            cmd.extend(['--model', model_name])
        if device != 'cpu':
            cmd.extend(['--device', device])
        if batch_size > 1:
            cmd.extend(['--batch-size', str(batch_size)])
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        if result.stdout:
            print(f"Piscis output: {result.stdout}")
        if result.stderr:
            print(f"Piscis warnings: {result.stderr}")
        
        # Load predictions
        # Piscis may save in different formats, try common ones
        if tmp_out_path.exists():
            predictions = np.load(tmp_out_path)
        else:
            # Try looking for other output files
            output_files = list(Path(tmp_output_dir).glob('*'))
            if output_files:
                # Try loading the first file found
                predictions = np.load(output_files[0])
            else:
                raise FileNotFoundError(f"Piscis output not found at {tmp_out_path}")
        
        # Clean up temp files
        Path(tmp_img_path).unlink()
        import shutil
        shutil.rmtree(tmp_output_dir)
        
        pred_time = time.time() - start_pred
        print(f"Detection time: {pred_time:.2f}s")
        
        # Process predictions
        # Piscis typically returns coordinates in (y, x) or (z, y, x) format
        # We need to convert to our (z, y, x) format
        if predictions.ndim == 2:
            if predictions.shape[1] == 2:
                # 2D coordinates (y, x) - need to add z dimension
                # For 3D images, we might need to process each slice or use max projection
                print("Warning: 2D predictions detected, converting to 3D...")
                # Assume spots are in middle z-slice or distribute across z
                z_mid = image.shape[0] // 2
                spots_3d = np.column_stack([
                    np.full(len(predictions), z_mid),
                    predictions[:, 0],  # y
                    predictions[:, 1]  # x
                ])
            elif predictions.shape[1] == 3:
                # Already 3D coordinates, but check order
                # Piscis might return (y, x, z) or (z, y, x)
                # Try to determine based on value ranges
                if np.max(predictions[:, 0]) < image.shape[1] and np.max(predictions[:, 1]) < image.shape[2]:
                    # Likely (y, x, z) - convert to (z, y, x)
                    spots_3d = predictions[:, [2, 0, 1]]
                else:
                    # Likely already (z, y, x)
                    spots_3d = predictions
            else:
                print(f"Warning: Unexpected prediction shape: {predictions.shape}")
                spots_3d = predictions
        else:
            print(f"Warning: Unexpected prediction format, shape: {predictions.shape}")
            spots_3d = predictions
        
        print(f"Detected {len(spots_3d)} spots")
        
        # Create spot plot
        plot_spot_size = config.get('plot_spot_size', 4)
        plot_inner_circle = config.get('plotInnerCircle', False)
        plot_spot_label = config.get('plotSpotLabel', False)
        
        print(f"\nCreating spot plot...")
        spot_plot = create_spot_plot(
            image,
            spots_3d,
            plot_spot_size=plot_spot_size,
            plot_inner_circle=plot_inner_circle,
            plot_spot_label=plot_spot_label
        )
        
        # Save results
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        image_name = Path(image_path).stem
        
        # Save spot plot
        spot_plot_path = output_dir / f"{image_name}_piscis_spotPlot.tif"
        imwrite(spot_plot_path, spot_plot, photometric='minisblack')
        print(f"Spot plot saved: {spot_plot_path.name}")
        
        # Save spots coordinates
        spots_path = output_dir / f"{image_name}_piscis_spots.npy"
        np.save(spots_path, spots_3d)
        print(f"Spots coordinates saved: {spots_path.name}")
        
        # Save summary
        summary_path = output_dir / f"{image_name}_piscis_summary.json"
        summary = {
            'image_path': str(image_path),
            'image_name': image_name,
            'image_shape': list(image.shape),
            'n_spots': len(spots_3d),
            'model_name': model_name if model_name else 'default',
            'timing': {
                'load_time': load_time,
                'prediction_time': pred_time,
                'total_time': load_time + pred_time
            },
            'parameters': {
                'batch_size': batch_size,
                'device': device,
                'plot_spot_size': plot_spot_size
            }
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved: {summary_path.name}")
        
        print(f"\n{'='*60}")
        print("Piscis detection completed!")
        print(f"{'='*60}")
        
        return spots_3d, summary
        
    except Exception as e:
        print(f"\nERROR during Piscis prediction: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    # Load config
    config_path = "config_piscis.yaml"
    
    if not Path(config_path).exists():
        print(f"ERROR: Config file not found: {config_path}")
        print("Please create config_piscis.yaml with image paths and parameters")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Get image path from config
    image_path = config.get('image_path')
    if not image_path:
        print("ERROR: 'image_path' not found in config_piscis.yaml")
        sys.exit(1)
    
    if not Path(image_path).exists():
        print(f"ERROR: Image file not found: {image_path}")
        sys.exit(1)
    
    # Get output directory
    output_dir = config.get('output_dir', Path(image_path).parent / 'piscis_results')
    
    # Run prediction
    predict_spots_piscis(image_path, output_dir, config)


if __name__ == "__main__":
    main()

