#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=48:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=Piscis3D_training
#SBATCH --output=piscis3d_training_output.log
#SBATCH --error=piscis3d_training_error.log

# Default parameters (can be overridden by command-line arguments)
MODEL_NAME=${1:-"spot_detection_3d_v1"}
DATASET_PATH=${2:-"/scratch/qgs8612/piscis_training_dataset_3d"}
OUTPUT_DIR=${3:-"/scratch/qgs8612/piscis3d_dataset"}
EPOCHS=${4:-100}
BATCH_SIZE=${5:-2}
LEARNING_RATE=${6:-0.001}

module load python-miniconda3
# Load JAX with GPU support via jax-fem module FIRST
module load jax-fem/0.0.8-gpu

# Initialize conda for bash shell and activate the environment
eval "$(conda shell.bash hook)"
conda activate smfish_env

# Use Python from the activated conda environment
PYTHON=$(which python)
echo "Using Python: $PYTHON"
echo "Python version: $($PYTHON --version)"

# Set PYTHONPATH to include Piscis3D
export PYTHONPATH="/home/qgs8612/planarian_smFISH/Piscis3D:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Check if JAX is available
echo "===== Checking JAX installation ====="
cd /tmp
JAX_TEST=$($PYTHON - <<'PYEOF' 2>&1
import sys
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

echo "$JAX_TEST"

if echo "$JAX_TEST" | grep -q "SUCCESS:"; then
    echo "JAX is installed and importable"
elif [ $JAX_EXIT -eq 0 ]; then
    echo "JAX check completed (exit code 0)"
else
    echo "ERROR: JAX not found in Python environment."
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
except Exception as e:
    print(f"Error checking JAX: {e}")
EOF

echo "============================================================"
echo "3D Piscis Model Training"
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
    echo "Please ensure the dataset has been generated using create_piscis_dataset_3d.py"
    exit 1
fi

# Run 3D Piscis training
cd /home/qgs8612/planarian_smFISH

# Run training script with parameters
$PYTHON tests/train_piscis_3d.py \
    --model_name "$MODEL_NAME" \
    --dataset_path "$DATASET_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --learning_rate "$LEARNING_RATE"

TRAINING_EXIT=$?

if [ $TRAINING_EXIT -eq 0 ]; then
    echo "============================================================"
    echo "Training script completed successfully!"
    echo "Model name: $MODEL_NAME"
    echo "Output directory: $OUTPUT_DIR"
    echo "============================================================"
else
    echo "============================================================"
    echo "Training script failed with exit code: $TRAINING_EXIT"
    echo "Check the error log for details: piscis3d_training_error.log"
    echo "============================================================"
    exit $TRAINING_EXIT
fi
