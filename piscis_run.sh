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

# Option A: source bashrc + activate (works if bashrc is set up)
# source ~/.bashrc
# conda activate /home/qgs8612/.conda/envs/smfish_env

# Option B: direct Python from environment (recommended)
PYTHON=/home/qgs8612/.conda/envs/smfish_env/bin/python

# GPU diagnostics
echo "===== NVIDIA-SMI ====="
nvidia-smi

echo "===== JAX GPU Check ====="
$PYTHON - <<EOF
try:
    import jax
    devices = jax.devices()
    print("JAX devices:", [str(d) for d in devices])
    gpu_devices = [d for d in devices if d.device_kind == 'gpu']
    if gpu_devices:
        print("GPU devices found:", [str(d) for d in gpu_devices])
        print("JAX GPU is available!")
    else:
        print("No GPU devices found, will use CPU")
except ImportError:
    print("JAX not installed")
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

