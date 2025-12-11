# run_server.py
import os
from pathlib import Path
import numpy as np
from tifffile import imread, imwrite

from functions.io_utils import create_folder_in_same_directory, load_config
from functions.spot_detection import detect_spots_real  # your real detection function


def run_pipeline(config_path: str):
    """
    Run the smFISH detection pipeline on the SSH/server environment using a YAML config.

    Args:
        config_path: Path to the YAML configuration file.
    """
    # --- Load config ---
    config = load_config(config_path)
    print("Loaded config parameters:")
    for k, v in config.items():
        print(f"{k}: {v}")

    # --- Create results folder ---
    smFISH_path = config["smFISHChannelPath"]
    results_folder = create_folder_in_same_directory(smFISH_path, "results")
    print(f"Results will be saved in: {results_folder}")

    # --- Load image ---
    img = imread(smFISH_path)
    print(f"Loaded image shape: {img.shape}, dtype: {img.dtype}")

    # --- Run smFISH detection ---
    spots, threshold = detect_spots_real(img, config, results_folder)

    # --- Save spots ---
    np.save(os.path.join(results_folder, "spots.npy"), spots)
    print(f"Saved detected spots: {spots.shape}")

    # --- Optionally save spot info ---
    if config.get("saveSpotInformation", True):
        info_path = os.path.join(results_folder, "spot_info.txt")
        with open(info_path, "w") as f:
            f.write(f"spots shape: {spots.shape}\n")
            f.write(f"threshold: {threshold}\n")
        print(f"Saved spot info at: {info_path}")

    print("✔ Pipeline completed successfully!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run smFISH detection on server")
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to the YAML configuration file"
    )
    args = parser.parse_args()

    run_pipeline(args.config)
