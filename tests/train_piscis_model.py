"""
Train Piscis model for spot detection.

This script:
1. Sets up the training directory structure
2. Trains a Piscis model using the generated dataset
3. Saves the trained model with checkpoints
"""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional, Tuple

try:
    import piscis
    from piscis import training as piscis_training
    HAS_PISCIS = True
except ImportError:
    HAS_PISCIS = False
    print("ERROR: Piscis is not installed.")
    print("Install with: pip install git+https://github.com/zjniu/Piscis.git")


def create_training_directory(base_dir: str = "/scratch/qgs8612/piscis_dataset"):
    """
    Create training directory structure.
    
    Parameters
    ----------
    base_dir : str
        Base directory for Piscis training outputs
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


def train_piscis_model(
    model_name: str,
    dataset_path: str,
    output_dir: str = "/scratch/qgs8612/piscis_dataset",
    initial_model_name: Optional[str] = None,
    adjustment: str = "standardize",
    input_size: Tuple[int, int] = (256, 256),
    random_seed: int = 0,
    batch_size: int = 4,
    learning_rate: float = 0.2,
    weight_decay: float = 0.0001,
    dropout_rate: float = 0.2,
    epochs: int = 400,
    warmup_fraction: float = 0.05,
    decay_fraction: float = 0.5,
    decay_transitions: int = 10,
    decay_factor: float = 0.5,
    dilation_iterations: int = 1,
    max_distance: float = 3.0,
    loss_weights: Optional[Dict[str, float]] = None,
    save_checkpoints: bool = True,
    channels: int = 1,
    verbose: bool = True
):
    """
    Train a Piscis spot detection model.
    
    Parameters
    ----------
    model_name : str
        Name of the model to train
    dataset_path : str
        Path to the directory containing training and validation datasets
    output_dir : str
        Directory where model outputs will be saved
    initial_model_name : Optional[str]
        Name of an existing model to initialize weights from (for fine-tuning)
    adjustment : str
        Image adjustment type: 'normalize' or 'standardize' (default: 'standardize')
    input_size : Tuple[int, int]
        Size of input images (default: (256, 256))
    random_seed : int
        Random seed for reproducibility (default: 0)
    batch_size : int
        Batch size for training (default: 4)
    learning_rate : float
        Learning rate for optimizer (default: 0.2)
    weight_decay : float
        Weight decay regularization strength (default: 0.0001)
    dropout_rate : float
        Dropout rate at skip connections (default: 0.2)
    epochs : int
        Number of training epochs (default: 400)
    warmup_fraction : float
        Fraction of epochs for learning rate warmup (default: 0.05)
    decay_fraction : float
        Fraction of epochs for learning rate decay (default: 0.5)
    decay_transitions : int
        Number of learning rate decay transitions (default: 10)
    decay_factor : float
        Multiplicative factor for each decay transition (default: 0.5)
    dilation_iterations : int
        Number of iterations to dilate ground truth labels (default: 1)
    max_distance : float
        Maximum distance for matching predicted and ground truth vectors (default: 3.0)
    loss_weights : Optional[Dict[str, float]]
        Weights for loss terms. If None, uses default {'l2': 0.25, 'smoothf1': 1.0}
    save_checkpoints : bool
        Whether to save checkpoints during training (default: True)
    channels : int
        Number of image channels (default: 1 for grayscale)
    verbose : bool
        Whether to print verbose output
    """
    if not HAS_PISCIS:
        raise ImportError("Piscis is not installed. Please install it first.")
    
    # Validate dataset path
    dataset_path_obj = Path(dataset_path)
    if not dataset_path_obj.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Create output directory
    output_dir_obj = create_training_directory(output_dir)
    
    # Validate warmup and decay fractions
    if warmup_fraction + decay_fraction > 1.0:
        raise ValueError(
            f"warmup_fraction ({warmup_fraction}) + decay_fraction ({decay_fraction}) "
            f"must be <= 1.0"
        )
    
    # Default loss weights if not provided
    if loss_weights is None:
        loss_weights = {'l2': 0.25, 'smoothf1': 1.0}
    
    if verbose:
        print(f"\n{'='*60}")
        print("Piscis Model Training Configuration")
        print(f"{'='*60}")
        print(f"Model name: {model_name}")
        print(f"Dataset path: {dataset_path}")
        print(f"Output directory: {output_dir}")
        print(f"Initial model: {initial_model_name if initial_model_name else 'None (training from scratch)'}")
        print(f"\nModel Parameters:")
        print(f"  Input size: {input_size}")
        print(f"  Channels: {channels}")
        print(f"  Adjustment: {adjustment}")
        print(f"  Dropout rate: {dropout_rate}")
        print(f"\nTraining Parameters:")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Weight decay: {weight_decay}")
        print(f"  Random seed: {random_seed}")
        print(f"\nLearning Rate Schedule:")
        print(f"  Warmup fraction: {warmup_fraction} ({int(warmup_fraction * epochs)} epochs)")
        print(f"  Decay fraction: {decay_fraction} ({int(decay_fraction * epochs)} epochs)")
        print(f"  Decay transitions: {decay_transitions}")
        print(f"  Decay factor: {decay_factor}")
        print(f"\nLoss Parameters:")
        print(f"  Dilation iterations: {dilation_iterations}")
        print(f"  Max distance: {max_distance}")
        print(f"  Loss weights: {loss_weights}")
        print(f"\nOther:")
        print(f"  Save checkpoints: {save_checkpoints}")
        print(f"{'='*60}\n")
    
    # Train the model
    print("Starting training...")
    try:
        piscis_training.train_model(
            model_name=model_name,
            dataset_path=dataset_path,
            initial_model_name=initial_model_name,
            adjustment=adjustment,
            input_size=input_size,
            random_seed=random_seed,
            batch_size=batch_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            dropout_rate=dropout_rate,
            epochs=epochs,
            warmup_fraction=warmup_fraction,
            decay_fraction=decay_fraction,
            decay_transitions=decay_transitions,
            decay_factor=decay_factor,
            dilation_iterations=dilation_iterations,
            max_distance=max_distance,
            loss_weights=loss_weights,
            save_checkpoints=save_checkpoints
        )
        
        print(f"\n{'='*60}")
        print(f"✓ Training completed successfully!")
        print(f"Model saved as: {model_name}")
        print(f"Output directory: {output_dir}")
        print(f"{'='*60}\n")
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"✗ Training failed with error:")
        print(f"{str(e)}")
        print(f"{'='*60}\n")
        raise


def main():
    """Command-line interface for Piscis model training."""
    parser = argparse.ArgumentParser(
        description="Train a Piscis spot detection model",
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
        help="Path to the directory containing training and validation datasets"
    )
    
    # Optional arguments
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/scratch/qgs8612/piscis_dataset",
        help="Directory where model outputs will be saved"
    )
    
    parser.add_argument(
        "--initial_model_name",
        type=str,
        default=None,
        help="Name of existing model to initialize weights from (for fine-tuning)"
    )
    
    parser.add_argument(
        "--adjustment",
        type=str,
        choices=["normalize", "standardize"],
        default="standardize",
        help="Image adjustment type"
    )
    
    parser.add_argument(
        "--input_size",
        type=int,
        nargs=2,
        default=[256, 256],
        metavar=("HEIGHT", "WIDTH"),
        help="Size of input images (height width)"
    )
    
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed for reproducibility"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for training"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.2,
        help="Learning rate for optimizer"
    )
    
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0001,
        help="Weight decay regularization strength"
    )
    
    parser.add_argument(
        "--dropout_rate",
        type=float,
        default=0.2,
        help="Dropout rate at skip connections"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=400,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--warmup_fraction",
        type=float,
        default=0.05,
        help="Fraction of epochs for learning rate warmup"
    )
    
    parser.add_argument(
        "--decay_fraction",
        type=float,
        default=0.5,
        help="Fraction of epochs for learning rate decay"
    )
    
    parser.add_argument(
        "--decay_transitions",
        type=int,
        default=10,
        help="Number of learning rate decay transitions"
    )
    
    parser.add_argument(
        "--decay_factor",
        type=float,
        default=0.5,
        help="Multiplicative factor for each decay transition"
    )
    
    parser.add_argument(
        "--dilation_iterations",
        type=int,
        default=1,
        help="Number of iterations to dilate ground truth labels"
    )
    
    parser.add_argument(
        "--max_distance",
        type=float,
        default=3.0,
        help="Maximum distance for matching predicted and ground truth vectors"
    )
    
    parser.add_argument(
        "--loss_weights",
        type=str,
        default=None,
        help="Loss weights as comma-separated key:value pairs (e.g., 'l2:0.25,smoothf1:1.0')"
    )
    
    parser.add_argument(
        "--no_checkpoints",
        action="store_true",
        help="Disable saving checkpoints during training"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose output"
    )
    
    args = parser.parse_args()
    
    # Parse loss weights if provided
    loss_weights = None
    if args.loss_weights:
        try:
            loss_weights = {}
            for pair in args.loss_weights.split(','):
                key, value = pair.split(':')
                loss_weights[key.strip()] = float(value.strip())
        except Exception as e:
            parser.error(f"Invalid loss_weights format: {e}. Expected format: 'key1:value1,key2:value2'")
    
    # Convert input_size to tuple
    input_size = tuple(args.input_size)
    
    # Train the model
    train_piscis_model(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        initial_model_name=args.initial_model_name,
        adjustment=args.adjustment,
        input_size=input_size,
        random_seed=args.random_seed,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        dropout_rate=args.dropout_rate,
        epochs=args.epochs,
        warmup_fraction=args.warmup_fraction,
        decay_fraction=args.decay_fraction,
        decay_transitions=args.decay_transitions,
        decay_factor=args.decay_factor,
        dilation_iterations=args.dilation_iterations,
        max_distance=args.max_distance,
        loss_weights=loss_weights,
        save_checkpoints=not args.no_checkpoints,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
