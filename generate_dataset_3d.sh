#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --mem=256G
# Note: Memory optimized with batch_size=5, overlap_factor=0.0
# If still running out of memory, try: --batch_size 1 --tile_size 16 128 128
#SBATCH --cpus-per-task=4
#SBATCH --job-name=Piscis3D_dataset_generation
#SBATCH --output=piscis3d_dataset_generation_output.log
#SBATCH --error=piscis3d_dataset_generation_error.log

# Default parameters (can be overridden by command-line arguments)
BASE_DIR=${1:-"/scratch/qgs8612/Experiment"}
OUTPUT_PATH=${2:-"/scratch/qgs8612/piscis_training_dataset_3d"}
WAVELENGTH=${3:-"565"}
TILE_SIZE_Z=${4:-32}
TILE_SIZE_Y=${5:-256}
TILE_SIZE_X=${6:-256}
MIN_SPOTS=${7:-1}
TRAIN_SIZE=${8:-0.7}
TEST_SIZE=${9:-0.15}
RANDOM_SEED=${10:-42}

module load python-miniconda3
# Load JAX with GPU support via jax-fem module FIRST
module load jax-fem/0.0.8-gpu

# Force JAX to use CPU for dataset generation (GPU not needed and CUDA libraries may not be available)
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

# Add Piscis3D to PYTHONPATH
export PYTHONPATH="$HOME/planarian_smFISH/Piscis3D:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Check if JAX is available
echo "===== Checking JAX installation ====="
cd /tmp
JAX_TEST=$($PYTHON - <<'PYEOF' 2>&1
import os
os.environ['JAX_PLATFORMS'] = 'cpu'
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
        import jax
        version = getattr(jax, '__version__', 'unknown')
        print(f'SUCCESS: JAX version {version} (with warnings)')
        sys.exit(0)
    except:
        print(f'FAILED: Exception - {e}', file=sys.stderr)
        sys.exit(1)
PYEOF
)

if [ $? -ne 0 ]; then
    echo "$JAX_TEST"
    echo "ERROR: JAX is not available!"
    exit 1
fi

echo "$JAX_TEST"
echo "JAX is available"

# Change to project directory
cd $HOME/planarian_smFISH

# Check that base directory exists
if [ ! -d "$BASE_DIR" ]; then
    echo "ERROR: Base directory not found: $BASE_DIR"
    exit 1
fi

echo "Base directory: $BASE_DIR"
echo "Output path: $OUTPUT_PATH"
echo "Wavelength: $WAVELENGTH"
echo "Tile size (z, y, x): ($TILE_SIZE_Z, $TILE_SIZE_Y, $TILE_SIZE_X)"
echo "Min spots: $MIN_SPOTS"

# Run the 3D dataset generation script
echo "============================================================"
echo "Starting 3D Piscis Dataset Generation"
echo "============================================================"

$PYTHON tests/create_piscis_dataset_3d.py \
    --base_dir "$BASE_DIR" \
    --output_path "$OUTPUT_PATH" \
    --wavelength "$WAVELENGTH" \
    --tile_size $TILE_SIZE_Z $TILE_SIZE_Y $TILE_SIZE_X \
    --min_spots $MIN_SPOTS \
    --train_size $TRAIN_SIZE \
    --test_size $TEST_SIZE \
    --random_seed $RANDOM_SEED \
    --overlap_factor 0.0 \
    --batch_size 5 \
    --exclude 0hr_Amputation 0hr_Incision

EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
    echo "============================================================"
    echo "3D Dataset generation completed successfully!"
    echo "Dataset saved to: $OUTPUT_PATH"
    echo "============================================================"
else
    echo "============================================================"
    echo "3D Dataset generation failed with exit code: $EXIT_CODE"
    echo "Check the error log for details: piscis3d_dataset_generation_error.log"
    echo "============================================================"
    exit $EXIT_CODE
fi
