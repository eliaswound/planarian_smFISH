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
# JAX is installed in the conda environment, so we don't need the jax-fem module
# Commented out to avoid conflicts - using JAX from conda environment instead
# module load jax-fem/0.0.8-gpu

# Initialize conda for bash shell and activate the environment
eval "$(conda shell.bash hook)"
conda activate smfish_env

# Use Python from the activated conda environment
PYTHON=$(which python)
echo "Using Python: $PYTHON"
echo "Python version: $($PYTHON --version)"

# Ensure user site-packages (where pip --user installs) are visible to this interpreter
# Include both 3.10 (preferred for this env) and 3.9 (in case prior installs were 3.9)
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH"
echo "PYTHONPATH set to include user site-packages: $PYTHONPATH"

# Check if JAX is available (JAX should already be installed in conda environment)
echo "===== Checking JAX installation ====="
# Test JAX import - capture both stdout and stderr, but only fail on ImportError
JAX_TEST=$($PYTHON - <<'PYEOF' 2>&1
import sys
try:
    import jax
    # If we get here, import succeeded
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
    echo "Please install JAX in your conda environment:"
    echo "    conda activate smfish_env"
    echo "    pip install \"jax[cuda12]\" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    exit 1
fi

# Check if Piscis is installed; if not, print a clear error (cannot auto-install because git is unavailable on compute nodes)
echo "===== Checking Piscis installation ====="
$PYTHON -c "import piscis" 2>/dev/null
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
    print("JAX should be installed in your conda environment")
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
$PYTHON /home/qgs8612/planarian_smFISH/piscis_test.py

