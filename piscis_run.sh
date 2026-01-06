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
# Load JAX with GPU support via jax-fem module
module load jax-fem/0.0.8-gpu

# Initialize conda for bash shell
eval "$(conda shell.bash hook)"

# Activate the conda environment
conda activate smfish_env

# Use Python from the activated conda environment
PYTHON=$(which python)
echo "Using Python: $PYTHON"

# Check if JAX is available, if not try to install it
echo "===== Checking JAX installation ====="
$PYTHON -c "import jax" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "JAX not found in Python environment, attempting to install..."
    echo "Note: jax-fem module is loaded, but JAX may need to be installed in your Python environment"
    $PYTHON -m pip install --user "jax[cuda12]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html || \
    $PYTHON -m pip install --user jax jaxlib || \
    echo "JAX installation failed, will try to continue..."
fi

# Check if Piscis is installed, if not try to install it
echo "===== Checking Piscis installation ====="
$PYTHON -c "import piscis" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Piscis not found, attempting to install..."
    $PYTHON -m pip install --user git+https://github.com/zjniu/Piscis.git || echo "Piscis installation failed, continuing anyway..."
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
    print("Make sure jax-fem module is loaded")
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

