"""
3D Training module for Piscis.
Adapted from the original 2D Piscis training code.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
from functools import partial
from jax import jit, random, value_and_grad
from flax import serialization
from flax.training import train_state
from pathlib import Path
from tqdm.auto import tqdm
from typing import Any, Dict, List, Optional, Tuple
import orbax.checkpoint as ocp

from piscis3d.models.spots_3d import SpotsModel3D, round_input_size_3d
from piscis3d.transforms import voronoi_transform_3d, batch_adjust_3d
from piscis3d.data_streaming import dataset_generator


class TrainState(train_state.TrainState):
    """TrainState for 3D model training."""
    batch_stats: Any
    key: jax.Array
    epoch: jax.Array


def create_train_state(
    key: jax.Array,
    input_size: Tuple[int, int, int],
    dropout_rate: float,
    channels: int,
    tx: optax.GradientTransformation,
    variables: Optional[Dict] = None
) -> TrainState:
    """Create a new TrainState object for 3D model.
    
    Note: This function cannot be JIT compiled because Flax model initialization
    involves creating model instances and calling init(), which returns mutable structures.
    """
    key, subkey = random.split(key, 2)
    
    # Initialize the model
    model = SpotsModel3D(dropout_rate=dropout_rate)
    
    # Initialize parameters
    if variables is None:
        # Create dummy input: (batch=1, z, y, x, channels)
        dummy_input = np.ones((1, *input_size, channels), dtype=np.float32)
        variables = model.init(subkey, dummy_input, train=False)
    
    # Create TrainState
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        batch_stats=variables['batch_stats'],
        key=key,
        epoch=jnp.array(-1, dtype=jnp.int32),
    )
    
    return state


def smoothf1_loss_3d(
    deltas_pred: jax.Array,
    labels_pred: jax.Array,
    deltas: jax.Array,
    labels: jax.Array,
    dilation_iterations: int,
    max_distance: float,
    epsilon: float = 1e-7
) -> jax.Array:
    """Compute SmoothF1 loss for 3D data.
    
    Adapted from 2D version to work with 3D arrays.
    """
    # Use labels as support for deltas_pred
    deltas_pred = labels * deltas_pred
    
    # Squeeze channel dimension
    labels_pred = labels_pred[:, :, :, :, 0]  # (batch, z, y, x)
    labels = labels[:, :, :, :, 0]  # (batch, z, y, x)
    
    # Compute distances between predicted and ground truth deltas
    distances = jnp.linalg.norm(deltas_pred - deltas, axis=-1)  # (batch, z, y, x)
    matches = jnp.maximum(1 - distances / max_distance, 0.0)
    
    # Simple matching: predicted label matches if within distance threshold
    # This is a simplified version - full implementation would use pooling
    tp = jnp.sum(labels_pred * labels * matches)
    fp = jnp.sum(labels_pred * (1 - labels))
    fn = jnp.sum((1 - labels_pred) * labels)
    
    # Compute SmoothF1 loss
    smoothf1 = -2 * tp / (2 * tp + fp + fn + epsilon)
    
    return smoothf1


def masked_l2_loss_3d(
    y_pred: jax.Array,
    y: jax.Array,
    mask: Optional[jax.Array] = None,
    epsilon: float = 1e-7
) -> jax.Array:
    """Compute masked L2 loss for 3D data."""
    if mask is None:
        mask = jnp.ones_like(y)
    
    diff = (y_pred - y) ** 2
    masked_diff = diff * mask
    loss = jnp.sum(masked_diff) / (jnp.sum(mask) + epsilon)
    
    return loss


@partial(jit, static_argnums=(4, 5, 7))
def loss_fn(
    params: Dict,
    state: TrainState,
    batch: Dict[str, jax.Array],
    key: Optional[jax.Array],
    dilation_iterations: int,
    max_distance: float,
    loss_weights: Dict[str, float],
    train: bool = True
) -> Tuple[jax.Array, Tuple[Dict[str, jax.Array], Optional[Dict]]]:
    """Compute loss for a batch."""
    images = batch['images']
    deltas = batch['deltas']
    labels = batch['labels']
    
    # Get model variables
    variables = {'params': params, 'batch_stats': state.batch_stats}
    
    # Forward pass
    if train and key is not None:
        deltas_pred, labels_pred, mutated_vars = state.apply_fn(
            variables, images, train=train, rngs={'dropout': key}, mutable=['batch_stats']
        )
    else:
        deltas_pred, labels_pred = state.apply_fn(variables, images, train=train)
        mutated_vars = None
    
    # Compute losses
    l2_loss = masked_l2_loss_3d(deltas_pred, deltas, labels)
    smoothf1 = smoothf1_loss_3d(
        deltas_pred, labels_pred, deltas, labels,
        dilation_iterations, max_distance
    )
    
    # Combined loss
    loss = loss_weights.get('l2', 0.25) * l2_loss + loss_weights.get('smoothf1', 1.0) * smoothf1
    
    metrics = {
        'loss': loss,
        'l2': l2_loss,
        'smoothf1': smoothf1
    }
    
    return loss, (metrics, mutated_vars)


@partial(jit, static_argnums=(3, 4))
def train_step(
    state: TrainState,
    batch: Dict[str, jax.Array],
    key: Optional[jax.Array],
    dilation_iterations: int,
    max_distance: float,
    loss_weights: Dict[str, float]
) -> Tuple[TrainState, Dict[str, jax.Array]]:
    """Perform a single training step."""
    grad_fn = value_and_grad(loss_fn, has_aux=True)
    
    (_, (metrics, mutated_vars)), grads = grad_fn(
        state.params, state, batch, key,
        dilation_iterations, max_distance, loss_weights, train=True
    )
    
    state = state.apply_gradients(grads=grads, batch_stats=mutated_vars['batch_stats'])
    
    return state, metrics


def prepare_batch(
    images: np.ndarray,
    coords: np.ndarray,
    tile_size: Tuple[int, int, int],
    dilation_iterations: int = 1
) -> Dict[str, jax.Array]:
    """Prepare a batch for training by applying Voronoi transform."""
    # Convert to JAX arrays
    images_list = []
    for img in images:
        img_array = np.asarray(img)
        if img_array.ndim == 3:  # (z, y, x)
            img_array = img_array[..., None]  # (z, y, x, 1)
        images_list.append(img_array)
    
    images_jax = jnp.array(images_list)  # (batch, z, y, x, channels)
    
    # Apply Voronoi transform to create ground truth
    labels_list = []
    deltas_list = []
    
    for coord in coords:
        coord_array = np.asarray(coord)
        if len(coord_array) == 0:
            # Empty coordinates - create empty labels and deltas
            labels_empty = np.zeros((*tile_size, 1), dtype=np.float32)
            deltas_empty = np.zeros((*tile_size, 3), dtype=np.float32)
            labels_list.append(labels_empty)
            deltas_list.append(deltas_empty)
        else:
            labels, deltas = voronoi_transform_3d(
                [coord_array],
                output_size=tile_size,
                dilation_iterations=dilation_iterations
            )
            labels_list.append(labels[0])
            deltas_list.append(deltas[0])
    
    # Stack into batches
    labels_batch = jnp.stack([jnp.array(l) for l in labels_list], axis=0)  # (batch, z, y, x, 1)
    deltas_batch = jnp.stack([jnp.array(d) for d in deltas_list], axis=0)  # (batch, z, y, x, 3)
    
    return {
        'images': images_jax,
        'labels': labels_batch,
        'deltas': deltas_batch
    }


def train_epoch(
    state: TrainState,
    dataset_dir: str,
    tile_size: Tuple[int, int, int],
    batch_size: int,
    epoch_learning_rate: float,
    dilation_iterations: int,
    max_distance: float,
    loss_weights: Dict[str, float],
    adjustment: Optional[str] = 'standardize',
    verbose: bool = True
) -> Tuple[TrainState, List[Dict[str, float]], Dict[str, float]]:
    """Train the model for a single epoch."""
    if verbose:
        print(f'Epoch {int(state.epoch) + 1}:')
    
    # Update learning rate (only works if optimizer was created with inject_hyperparams)
    # Check if hyperparams exist before trying to update
    if hasattr(state.opt_state, 'hyperparams') and 'learning_rate' in state.opt_state.hyperparams:
        state.opt_state.hyperparams['learning_rate'] = jnp.array(epoch_learning_rate, dtype=jnp.float32)
    # If hyperparams don't exist, we'll just use the fixed learning rate
    # (this means we need to recreate the optimizer with the new LR, but that's complex)
    # For now, we'll assume inject_hyperparams is used
    
    # Get random key for this epoch
    key = random.fold_in(state.key, state.epoch)
    
    # Create dataset generator
    gen = dataset_generator(dataset_dir, split='train', shuffle_batches=True, rng_key=key)
    
    batch_metrics = []
    n_batches = 0
    
    # Process batches
    pbar = tqdm(total=None, disable=not verbose) if verbose else None
    
    for images_batch, coords_batch in gen:
        # Limit batch size
        if len(images_batch) > batch_size:
            images_batch = images_batch[:batch_size]
            coords_batch = coords_batch[:batch_size]
        
        if len(images_batch) == 0:
            continue
        
        # Adjust images
        if adjustment:
            images_batch = batch_adjust_3d(images_batch, adjustment)
        
        # Prepare batch
        try:
            batch = prepare_batch(images_batch, coords_batch, tile_size, dilation_iterations)
        except Exception as e:
            if verbose:
                print(f"Warning: Failed to prepare batch: {e}")
            continue
        
        # Split key for dropout
        key, subkey = random.split(key)
        
        # Training step
        state, metrics = train_step(
            state, batch, subkey, dilation_iterations, max_distance, loss_weights
        )
        
        metrics = {k: float(v) for k, v in metrics.items()}
        batch_metrics.append(metrics)
        n_batches += 1
        
        # Update progress bar
        if pbar:
            epoch_metrics = {
                k: np.mean([m[k] for m in batch_metrics]).astype(float)
                for k in batch_metrics[0]
            }
            summary = (
                f"(train) loss: {epoch_metrics['loss']:>6.4f}, "
                f"l2: {epoch_metrics['l2']:>6.4f}, "
                f"smoothf1: {epoch_metrics['smoothf1']:>6.4f}"
            )
            pbar.update(1)
            pbar.set_postfix_str(summary)
    
    if pbar:
        pbar.close()
    
    # Compute epoch metrics
    if batch_metrics:
        epoch_metrics = {
            k: np.mean([m[k] for m in batch_metrics]).astype(float)
            for k in batch_metrics[0]
        }
        epoch_metrics['n_batches'] = n_batches
    else:
        epoch_metrics = {}
    
    epoch_metrics['learning_rate'] = epoch_learning_rate
    
    # Update epoch
    state = state.replace(epoch=state.epoch + 1)
    
    return state, batch_metrics, epoch_metrics


def compute_learning_rate_schedule(
    epoch: int,
    epochs: int,
    learning_rate: float,
    warmup_fraction: float,
    decay_fraction: float,
    decay_transitions: int,
    decay_factor: float
) -> float:
    """Compute learning rate for current epoch."""
    warmup_epochs = int(epochs * warmup_fraction)
    decay_start = int(epochs * (1 - decay_fraction))
    
    if epoch < warmup_epochs:
        # Warmup: linear increase
        return learning_rate * (epoch + 1) / warmup_epochs
    elif epoch >= decay_start:
        # Decay: step decay
        decay_epochs = epoch - decay_start
        decay_steps = (epochs - decay_start) // decay_transitions
        decay_count = min(decay_epochs // decay_steps, decay_transitions)
        return learning_rate * (decay_factor ** decay_count)
    else:
        # Constant
        return learning_rate


def train_model(
    model_name: str,
    dataset_path: str,
    output_dir: str,
    tile_size: Tuple[int, int, int] = (8, 64, 64),
    random_seed: int = 42,
    batch_size: int = 2,
    learning_rate: float = 0.001,
    weight_decay: float = 1e-4,
    dropout_rate: float = 0.2,
    epochs: int = 100,
    warmup_fraction: float = 0.05,
    decay_fraction: float = 0.5,
    decay_transitions: int = 10,
    decay_factor: float = 0.5,
    dilation_iterations: int = 1,
    max_distance: float = 3.0,
    loss_weights: Optional[Dict[str, float]] = None,
    adjustment: Optional[str] = 'standardize',
    save_checkpoints: bool = True,
    verbose: bool = True
) -> None:
    """Train a 3D Piscis model.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    dataset_path : str
        Path to streaming dataset directory
    output_dir : str
        Output directory for models and checkpoints
    tile_size : Tuple[int, int, int]
        Tile size (z, y, x)
    random_seed : int
        Random seed
    batch_size : int
        Batch size
    learning_rate : float
        Initial learning rate
    weight_decay : float
        Weight decay
    dropout_rate : float
        Dropout rate
    epochs : int
        Number of epochs
    warmup_fraction : float
        Fraction of epochs for warmup
    decay_fraction : float
        Fraction of epochs for decay
    decay_transitions : int
        Number of decay transitions
    decay_factor : float
        Decay factor
    dilation_iterations : int
        Dilation iterations for ground truth
    max_distance : float
        Max distance for matching
    loss_weights : Optional[Dict[str, float]]
        Loss weights
    adjustment : Optional[str]
        Image adjustment type
    save_checkpoints : bool
        Whether to save checkpoints
    verbose : bool
        Verbose output
    """
    if loss_weights is None:
        loss_weights = {'l2': 0.25, 'smoothf1': 1.0}
    
    # Round input size
    tile_size = round_input_size_3d(tile_size)
    
    if verbose:
        print(f"Rounded tile size: {tile_size}")
    
    # Create random key
    key = random.PRNGKey(random_seed)
    
    # Create optimizer with inject_hyperparams to allow learning rate updates
    # This allows us to update the learning rate during training
    base_optimizer = partial(optax.adamw, weight_decay=weight_decay)
    tx = optax.inject_hyperparams(base_optimizer)(learning_rate=learning_rate)
    
    # Create training state
    if verbose:
        print("Initializing model...")
    state = create_train_state(key, tile_size, dropout_rate, channels=1, tx=tx)
    
    # Create output directories
    output_path = Path(output_dir)
    models_dir = output_path / "models"
    checkpoints_dir = output_path / "checkpoints"
    models_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    if verbose:
        print(f"\nStarting training for {epochs} epochs...")
        print(f"Dataset: {dataset_path}")
        print(f"Output: {output_dir}")
        print(f"Model: {model_name}\n")
    
    for epoch in range(epochs):
        # Compute learning rate
        lr = compute_learning_rate_schedule(
            epoch, epochs, learning_rate, warmup_fraction,
            decay_fraction, decay_transitions, decay_factor
        )
        
        # Train epoch
        state, batch_metrics, epoch_metrics = train_epoch(
            state, dataset_path, tile_size, batch_size, lr,
            dilation_iterations, max_distance, loss_weights,
            adjustment, verbose
        )
        
        # Save checkpoint
        if save_checkpoints and (epoch + 1) % 10 == 0:
            checkpoint_path = checkpoints_dir / f"{model_name}_epoch_{epoch+1}"
            # Save using orbax
            ckpt = {
                'params': state.params,
                'batch_stats': state.batch_stats,
                'opt_state': state.opt_state,
                'epoch': state.epoch,
            }
            ocp.CheckpointManager(checkpoint_path).save(epoch, ckpt)
            if verbose:
                print(f"Saved checkpoint: {checkpoint_path}")
    
    # Save final model
    final_model_path = models_dir / model_name
    final_ckpt = {
        'params': state.params,
        'batch_stats': state.batch_stats,
        'opt_state': state.opt_state,
        'epoch': state.epoch,
    }
    # Save using orbax checkpoint manager
    ckpt_manager = ocp.CheckpointManager(final_model_path)
    ckpt_manager.save(epochs, final_ckpt)
    
    if verbose:
        print(f"\n✓ Training completed!")
        print(f"Model saved to: {final_model_path}")
