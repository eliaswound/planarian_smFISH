#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=Piscis_training
#SBATCH --output=piscis_training_output.log
#SBATCH --error=piscis_training_error.log

# Default parameters (can be overridden by command-line arguments)
MODEL_NAME=${1:-"spot_detection_v1"}
DATASET_PATH=${2:-"/scratch/qgs8612/piscis_training_dataset"}
OUTPUT_DIR=${3:-"/scratch/qgs8612/piscis_dataset"}
EPOCHS=${4:-400}
BATCH_SIZE=${5:-4}
LEARNING_RATE=${6:-0.2}

module load python-miniconda3
# Load JAX with GPU support via jax-fem module FIRST
# This sets up environment variables that make JAX available
module load jax-fem/0.0.8-gpu

# Initialize conda for bash shell and activate the environment
eval "$(conda shell.bash hook)"
conda activate smfish_env

# Use Python from the activated conda environment
PYTHON=$(which python)
echo "Using Python: $PYTHON"
echo "Python version: $($PYTHON --version)"

# The jax-fem module should have set up PYTHONPATH or other env vars
echo "PYTHONPATH: $PYTHONPATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Check if JAX is available (should come from jax-fem module)
echo "===== Checking JAX installation ====="
# Change to a clean directory to avoid importing from current directory
cd /tmp
JAX_TEST=$($PYTHON - <<'PYEOF' 2>&1
import sys
# Remove current directory from sys.path to avoid importing local modules
if '' in sys.path:
    sys.path.remove('')
if '.' in sys.path:
    sys.path.remove('.')
try:
    import jax
    version = getattr(jax, '__version__', 'unknown')
    print(f'SUCCESS: JAX version {version}')
    sys.exit(0)
except ImportError as e:
    print(f'FAILED: ImportError - {e}', file=sys.stderr)
    sys.exit(1)
except Exception as e:
    # Other exceptions (like CUDA errors) are OK - import succeeded
    try:
        version = getattr(jax, '__version__', 'unknown')
        print(f'SUCCESS: JAX version {version} (with warnings)')
        sys.exit(0)
    except:
        print(f'FAILED: Exception during import - {e}', file=sys.stderr)
        sys.exit(1)
PYEOF
)
JAX_EXIT=$?

# Show the output
echo "$JAX_TEST"

# Check if we got success message
if echo "$JAX_TEST" | grep -q "SUCCESS:"; then
    echo "JAX is installed and importable"
elif [ $JAX_EXIT -eq 0 ]; then
    # Exit code 0 but no SUCCESS message - might have worked anyway
    echo "JAX check completed (exit code 0)"
else
    echo "ERROR: JAX not found in Python environment."
    echo "The jax-fem module should provide JAX. Check if module is loaded correctly."
    exit 1
fi

# Check if Piscis is installed; if not, print a clear error (cannot auto-install because git is unavailable on compute nodes)
echo "===== Checking Piscis installation ====="
cd /tmp
$PYTHON -c "import sys; sys.path = [p for p in sys.path if p not in ('', '.')]; import piscis" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "ERROR: Piscis is not installed in $PYTHON."
    echo "Please log into a login node, activate smfish_env, and run:"
    echo "    pip install git+https://github.com/zjniu/Piscis.git"
    exit 1
fi

# GPU diagnostics
echo "===== NVIDIA-SMI ====="
nvidia-smi

echo "===== JAX GPU Check ====="
cd /tmp
$PYTHON - <<EOF
try:
    import jax
    print("JAX version:", jax.__version__ if hasattr(jax, '__version__') else 'unknown')
    devices = jax.devices()
    print("JAX devices:", [str(d) for d in devices])
    gpu_devices = [d for d in devices if d.device_kind == 'gpu']
    if gpu_devices:
        print("GPU devices found:", [str(d) for d in gpu_devices])
        print("JAX GPU is available!")
    else:
        print("No GPU devices found, will use CPU")
except ImportError as e:
    print(f"JAX not installed: {e}")
    print("JAX should be provided by the jax-fem module")
except Exception as e:
    print(f"Error checking JAX: {e}")
EOF

echo "===== CUDA visible devices ====="
$PYTHON - <<EOF
import os
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
print("CUDA_VISIBLE_DEVICES:", cuda_visible)
EOF

echo "============================================================"
echo "Piscis Model Training"
echo "============================================================"
echo "Model name: $MODEL_NAME"
echo "Dataset path: $DATASET_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Epochs: $EPOCHS"
echo "Batch size: $BATCH_SIZE"
echo "Learning rate: $LEARNING_RATE"
echo "============================================================"

# Verify dataset path exists
if [ ! -d "$DATASET_PATH" ]; then
    echo "ERROR: Dataset path not found: $DATASET_PATH"
    echo "Please ensure the dataset has been generated using create_piscis_dataset.py"
    exit 1
fi

# Run Piscis training
# Change to script directory to ensure relative imports work
cd /home/qgs8612/planarian_smFISH

# Run training script with parameters
$PYTHON tests/train_piscis_model.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE"

TRAINING_EXIT=$?

if [ $TRAINING_EXIT -eq 0 ]; then
    echo "============================================================"
    echo "Training completed successfully!"
    echo "Model saved as: $MODEL_NAME"
    echo "Output directory: $OUTPUT_DIR"
    echo "============================================================"
else
    echo "============================================================"
    echo "Training failed with exit code: $TRAINING_EXIT"
    echo "Check the error log for details: piscis_training_error.log"
    echo "============================================================"
    exit $TRAINING_EXIT
fi
