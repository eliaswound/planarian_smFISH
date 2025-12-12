# run_server.py
"""
Server mode entry point for smFISH detection pipeline.
Author: Elias Guan
"""



import os
from functions.io_utils import load_config, create_folder_in_same_directory
from functions.spot_detection import detect_spots_from_config
import numpy as np
from tifffile import imwrite


def main():
    # Step 1: Load config
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)
    print("Loaded config parameters:")
    for key, value in config.items():
        print(f"{key}: {value}")

    # Step 2: Create main results folder
    exp_path = config.get("smFISHChannelPath")
    results_folder = create_folder_in_same_directory(exp_path, "results")

    # Step 3: Create subfolders for organized outputs
    npy_folder = create_folder_in_same_directory(results_folder, "npy")
    tiff_folder = create_folder_in_same_directory(results_folder, "tiff")
    plots_folder = create_folder_in_same_directory(results_folder, "plots")


    # Step 4: Run spot detection (control + experiment)
    spots_exp, threshold_used, img_log_exp = detect_spots_from_config(
        config, results_folder=results_folder
    )

    # Step 5: Save experiment results
    # Save spots as npy
    np.save(os.path.join(npy_folder, "spots_exp.npy"), spots_exp)

    # Save LoG filtered image as tiff
    imwrite(os.path.join(tiff_folder, "smFISH_LoG_filtered.tif"), img_log_exp, photometric='minisblack')

    print(f"Experiment spots detected: {len(spots_exp)}")
    print(f"Threshold used for experiment: {threshold_used}")
    print(f"Results saved in:\n  {npy_folder}\n  {tiff_folder}\n  {plots_folder}")


if __name__ == "__main__":
    main()
