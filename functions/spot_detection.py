
"""
Spot detection functions for smFISH pipeline (server mode)
Author: Elias Guan
"""

from tifffile import imread, imwrite
from bigfish.stack import log_filter
from bigfish.detection import detect_spots as bf_detect_spots
import numpy as np
import os


def detect_control_spots(control_path, config, results_folder):
    """
    Run BigFISH spot detection on control image with threshold=0
    to get all candidate spots. This is used to set an appropriate
    threshold for the real experiment image.

    Args:
        control_path (str): Path to control image
        config (dict): Configuration parameters
        results_folder (str): Folder to save results

    Returns:
        np.ndarray: All candidate spots detected on the control image
    """
    # Load image
    img = imread(control_path)
    print(f"Loaded control image: {img.shape}")

    # Apply LoG filter
    kernel_size = config["kernel_size"]
    img_log = log_filter(img, kernel_size)
    imwrite(os.path.join(results_folder, "control_LoG_filtered.tif"), img_log)

    # Detect all spots with threshold=0
    spots_all, _ = bf_detect_spots(
        images=img,
        threshold=0,
        return_threshold=True,
        voxel_size=config["voxel_size"],
        spot_radius=config["spot_size"],
        log_kernel_size=kernel_size,
        minimum_distance=config["minimal_distance"]
    )

    print(f"Detected {len(spots_all)} candidate spots on control image (threshold=0).")
    return spots_all


def find_spots_around(coordinate, array, max_iterations=10):
    """
    Flood-fill to find all voxels belonging to a 3D spot starting from a coordinate.
    Uses 6-connectivity (faces only) and stops at zero intensity or max_iterations.

    Args:
        coordinate (tuple/list/array): (z, y, x) starting voxel
        array (np.ndarray): 3D LoG-filtered image
        max_iterations (int): Maximum iterations to avoid infinite loops

    Returns:
        np.ndarray: Array of voxel coordinates belonging to the spot
    """
    # 6-connectivity vectors
    vectors = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ], dtype=int)

    outer_edge = [np.array(coordinate, dtype=np.int16)]
    spots_collection = [np.array(coordinate, dtype=np.int16)]
    shape = array.shape

    for _ in range(max_iterations):
        new_outer_edge = []
        for item in outer_edge:
            for vec in vectors:
                neighbor = item + vec
                # Skip out-of-bounds neighbors
                if np.any(neighbor < 0) or np.any(neighbor >= shape):
                    continue
                # Skip already visited
                if any(np.array_equal(neighbor, v) for v in spots_collection):
                    continue
                # Add neighbor if intensity > 0 and decreasing from center
                if array[tuple(neighbor)] > 0 and array[tuple(item)] >= array[tuple(neighbor)]:
                    spots_collection.append(neighbor)
                    new_outer_edge.append(neighbor)
        if len(new_outer_edge) == 0:
            break
        outer_edge = new_outer_edge

    return np.array(spots_collection, dtype=np.int16)





def compute_control_spot_intensities(control_path, config, results_folder):
    """
    Compute intensities of candidate spots on the control image.

    Args:
        control_path (str): path to control image
        config (dict): configuration parameters
        results_folder (str): folder to save intermediate results

    Returns:
        spots_all (list): list of all detected spot coordinates
        spot_intensities (np.array): intensity of each spot
        img_log (np.array): LoG filtered image
    """
    if not os.path.exists(control_path):
        raise FileNotFoundError(f"Control image not found: {control_path}")

    # Step 1: Load image
    img = imread(control_path)
    print(f"Loaded control image: {img.shape}")

    # Step 2: LoG filtering
    kernel_size = config["kernel_size"]
    img_log = log_filter(img, kernel_size)
    imwrite(os.path.join(results_folder, "control_LoG_filtered.tif"), img_log)
    print("Saved LoG filtered control image.")

    # Step 3: BigFISH detection at threshold = 0
    spots_all, _ = bf_detect_spots(
        images=img,
        threshold=0,
        return_threshold=True,
        voxel_size=config["voxel_size"],
        spot_radius=config["spot_size"],
        log_kernel_size=kernel_size,
        minimum_distance=config["minimal_distance"]
    )
    print(f"Detected {len(spots_all)} candidate spots at threshold=0.")

    # Step 4 & 5: Compute intensity for each spot
    spot_intensities = []
    for spot in spots_all:
        coords = find_spots_around(spot, img_log)
        intensity = np.sum(img_log[tuple(coords.T)])  # sum of voxel values
        spot_intensities.append(intensity)

    spot_intensities = np.array(spot_intensities)
    print(f"Computed intensities for {len(spot_intensities)} spots.")

    return spots_all, spot_intensities, img_log


def compute_control_threshold(spots_all, spot_intensities, percentile=0.99):
    """
    Compute threshold for experimental image based on control spots.

    Args:
        spots_all (list): output from BigFISH detect_spots (s[3] is intensity)
        spot_intensities (np.array): intensities computed from find_spots_around
        percentile (float): fraction of spots to keep below threshold (0.99 = 99%)

    Returns:
        threshold (float): maximum s[3] of the selected control spots
        selected_indices (np.array): indices of spots used
    """
    # Step 1: number of spots to select
    n_spots = int(np.floor(len(spot_intensities) * percentile))

    # Step 2: indices of smallest intensities
    sorted_indices = np.argsort(spot_intensities)
    selected_indices = sorted_indices[:n_spots]

    # Step 3: map back to spots_all and get s[3] values
    selected_spot_s3 = [spots_all[i][3] for i in selected_indices]

    # Step 4: maximum of these s[3] values → threshold
    threshold = max(selected_spot_s3)

    return threshold, selected_indices

# functions/spot_detection.py

def detect_spots_from_config(config):
    """
    Main function to detect spots on experiment image using control image threshold or config threshold.

    Args:
        config (dict): Configuration dictionary

    Returns:
        spots_exp (np.ndarray): Detected experiment spots
        exp_threshold_used (float): Threshold applied
    """
    results_folder = os.path.join(os.path.dirname(config["smFISHChannelPath"]), "results")
    os.makedirs(results_folder, exist_ok=True)

    control_threshold = None

    # Step 1: Control image threshold
    if config.get("controlImage") and config.get("controlPath"):
        print("Running control image detection...")
        from .spot_detection import compute_control_spot_intensities, compute_control_threshold
        spots_all, spot_intensities, img_log = compute_control_spot_intensities(
            config["controlPath"], config, results_folder
        )
        control_threshold, selected_indices = compute_control_threshold(
            spots_all, spot_intensities, percentile=0.99
        )
        print(f"Control threshold computed: {control_threshold}")

    # Step 2: Load experiment image
    img_exp = imread(config["smFISHChannelPath"])
    print(f"Loaded experiment image: {img_exp.shape}")

    # Step 3: Determine threshold
    if control_threshold is not None:
        threshold_to_use = control_threshold
        print("Using threshold from control image.")
    elif config.get("experimentAverageThreshold") is not None:
        threshold_to_use = config["experimentAverageThreshold"]
        print(f"Using threshold from config: {threshold_to_use}")
    else:
        threshold_to_use = None  # BigFISH auto-threshold
        print("Using BigFISH automatic threshold.")

    # Step 4: Detect experiment spots
    spots_exp, exp_threshold_used = bf_detect_spots(
        images=img_exp,
        threshold=threshold_to_use,
        return_threshold=True,
        voxel_size=config["voxel_size"],
        spot_radius=config["spot_size"],
        log_kernel_size=config["kernel_size"],
        minimum_distance=config["minimal_distance"],
    )
    print(f"Detected {len(spots_exp)} spots on experiment image (threshold={exp_threshold_used})")

    # Step 5: Save LoG filtered image and spots
    img_log_exp = log_filter(img_exp, config["kernel_size"])
    imwrite(os.path.join(results_folder, "experiment_LoG_filtered.tif"), img_log_exp)
    np.save(os.path.join(results_folder, "experiment_spots.npy"), spots_exp)
    print("Saved experiment LoG filtered image and spots array.")

    return spots_exp, exp_threshold_used
