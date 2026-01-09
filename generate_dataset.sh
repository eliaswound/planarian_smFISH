#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --job-name=Piscis_dataset_generation
#SBATCH --output=piscis_dataset_generation_output.log
#SBATCH --error=piscis_dataset_generation_error.log

# Default parameters (can be overridden by command-line arguments)
BASE_DIR=${1:-"/scratch/qgs8612/Experiment"}
OUTPUT_PATH=${2:-"/scratch/qgs8612/piscis_training_dataset"}
WAVELENGTH=${3:-"565"}
TILE_SIZE_HEIGHT=${4:-256}
TILE_SIZE_WIDTH=${5:-256}
MIN_SPOTS=${6:-1}
TRAIN_SIZE=${7:-0.7}
TEST_SIZE=${8:-0.15}
RANDOM_SEED=${9:-42}

module load python-miniconda3
# Load JAX with GPU support via jax-fem module FIRST
# This sets up environment variables that make JAX available
module load jax-fem/0.0.8-gpu

# Force JAX to use CPU for dataset generation (GPU not needed and CUDA libraries may not be available)
# Dataset generation is mostly I/O and data preparation, CPU is sufficient
# Set this AFTER loading jax-fem module to override any GPU settings
export JAX_PLATFORMS=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Initialize conda for bash shell and activate the environment
eval "$(conda shell.bash hook)"
conda activate smfish_env

# Use Python from the activated conda environment
PYTHON=$(which python)
echo "Using Python: $PYTHON"
echo "Python version: $($PYTHON --version)"
echo "JAX_PLATFORMS: $JAX_PLATFORMS (forced to CPU for dataset generation)"

# The jax-fem module should have set up PYTHONPATH or other env vars
echo "PYTHONPATH: $PYTHONPATH"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"

# Check if JAX is available (should come from jax-fem module)
echo "===== Checking JAX installation ====="
# Change to a clean directory to avoid importing from current directory
cd /tmp
JAX_TEST=$($PYTHON - <<'PYEOF' 2>&1
import os
import sys
# Force CPU mode for JAX check
os.environ['JAX_PLATFORMS'] = 'cpu'
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

# GPU diagnostics (optional for dataset generation, but good to check)
echo "===== NVIDIA-SMI ====="
nvidia-smi

echo "===== JAX Device Check (CPU mode) ====="
cd /tmp
$PYTHON - <<EOF
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
try:
    import jax
    print("JAX version:", jax.__version__ if hasattr(jax, '__version__') else 'unknown')
    devices = jax.devices()
    print("JAX devices:", [str(d) for d in devices])
    print("Using CPU mode for dataset generation (GPU not needed)")
except ImportError as e:
    print(f"JAX not installed: {e}")
    print("JAX should be provided by the jax-fem module")
except Exception as e:
    print(f"Error checking JAX: {e}")
    print("Continuing with CPU mode...")
EOF

echo "===== CUDA visible devices ====="
$PYTHON - <<EOF
import os
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')
print("CUDA_VISIBLE_DEVICES:", cuda_visible)
EOF

echo "============================================================"
echo "Piscis Dataset Generation"
echo "============================================================"
echo "Base directory: $BASE_DIR"
echo "Output path: $OUTPUT_PATH"
echo "Wavelength: $WAVELENGTH"
echo "Tile size: ${TILE_SIZE_HEIGHT}x${TILE_SIZE_WIDTH}"
echo "Min spots per tile: $MIN_SPOTS"
echo "Train size: $TRAIN_SIZE"
echo "Test size: $TEST_SIZE"
echo "Validation size: $(echo "1 - $TRAIN_SIZE - $TEST_SIZE" | bc)"
echo "Random seed: $RANDOM_SEED"
echo "============================================================"

# Verify base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "ERROR: Base directory not found: $BASE_DIR"
    echo "Please check the path and ensure it exists."
    exit 1
fi

# Check for expected subdirectories
echo "Checking for expected condition directories..."
EXPECTED_CONDITIONS=("6hr_Incision" "6hr_Amputation" "12hr_Incision" "12hr_Amputation")
FOUND_CONDITIONS=()
for condition in "${EXPECTED_CONDITIONS[@]}"; do
    if [ -d "$BASE_DIR/$condition" ]; then
        FOUND_CONDITIONS+=("$condition")
        echo "  ✓ Found: $condition"
    else
        echo "  ✗ Missing: $condition"
    fi
done

if [ ${#FOUND_CONDITIONS[@]} -eq 0 ]; then
    echo "ERROR: No expected condition directories found in $BASE_DIR"
    echo "Expected at least one of: ${EXPECTED_CONDITIONS[*]}"
    exit 1
fi

echo "Found ${#FOUND_CONDITIONS[@]} condition directory(ies)"

# Run dataset generation script
# Change to script directory to ensure relative imports work
cd /home/qgs8612/planarian_smFISH

# Run dataset generation with parameters
# Note: Default exclude list ['0hr_Amputation', '0hr_Incision'] won't affect 6hr/12hr conditions
$PYTHON tests/create_piscis_dataset.py \
    --base_dir "$BASE_DIR" \
    --output_path "$OUTPUT_PATH" \
    --wavelength "$WAVELENGTH" \
    --tile_size "$TILE_SIZE_HEIGHT" "$TILE_SIZE_WIDTH" \
    --min_spots "$MIN_SPOTS" \
    --train_size "$TRAIN_SIZE" \
    --test_size "$TEST_SIZE" \
    --random_seed "$RANDOM_SEED"

DATASET_EXIT=$?

if [ $DATASET_EXIT -eq 0 ]; then
    echo "============================================================"
    echo "Dataset generation completed successfully!"
    echo "Dataset saved at: $OUTPUT_PATH"
    echo "============================================================"
    echo ""
    echo "Next steps:"
    echo "1. Verify the dataset was created correctly"
    echo "2. Run training with:"
    echo "   sbatch piscis_train.sh [model_name] $OUTPUT_PATH [output_dir] [epochs] [batch_size] [learning_rate]"
    echo "============================================================"
else
    echo "============================================================"
    echo "Dataset generation failed with exit code: $DATASET_EXIT"
    echo "Check the error log for details: piscis_dataset_generation_error.log"
    echo "============================================================"
    exit $DATASET_EXIT
fi
