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

# Use the Python interpreter from your smfish_env conda environment (where Piscis is installed)
PYTHON=/home/qgs8612/.conda/envs/smfish_env/bin/python
echo "Using Python: $PYTHON"

# Ensure user site-packages (where pip --user installs) are visible to this interpreter
# Include both 3.10 (preferred for this env) and 3.9 (in case prior installs were 3.9)
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH"
echo "PYTHONPATH set to include user site-packages: $PYTHONPATH"

# Check if JAX is available (JAX should already be installed in conda environment)
# Use a Python script that properly handles import errors vs CUDA warnings
echo "===== Checking JAX installation ====="
$PYTHON - <<'PYEOF' 2>/dev/null
import sys
try:
    import jax
    print(f'JAX version: {jax.__version__}')
    sys.exit(0)  # Success
except ImportError:
    print("ERROR: JAX not found in Python environment.", file=sys.stderr)
    sys.exit(1)  # Import failed
except Exception as e:
    # JAX imported but other errors (like CUDA warnings) - that's OK
    print(f'JAX version: {jax.__version__}')
    sys.exit(0)  # Success (import worked, just warnings)
PYEOF

if [ $? -ne 0 ]; then
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

