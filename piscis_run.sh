#!/bin/bash
#SBATCH --account=b1042
#SBATCH --partition=genomics-gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --job-name=Piscis_spot_detection
#SBATCH --output=piscis_output.log

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
# Let's check what it added
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

# Run Piscis spot detection
# Change to script directory to ensure relative imports work
cd /home/qgs8612/planarian_smFISH
$PYTHON /home/qgs8612/planarian_smFISH/piscis_test.py
