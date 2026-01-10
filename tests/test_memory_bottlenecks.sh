#!/bin/bash
#SBATCH --job-name=memory_profile
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --output=memory_profile_%j.log
#SBATCH --error=memory_profile_%j.err

# Memory profiling script for dataset generation
# This helps identify memory bottlenecks

echo "=========================================="
echo "Memory Profiling for 3D Dataset Generation"
echo "=========================================="
echo "Date: $(date)"
echo "Node: $(hostname)"
echo ""

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate smfish_env

# Set Python path
export PYTHONPATH="$HOME/.local/lib/python3.10/site-packages:$HOME/.local/lib/python3.9/site-packages:$PYTHONPATH"
export JAX_PLATFORMS=cpu

# Check if psutil is available
python3 -c "import psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing psutil for memory profiling..."
    pip install --user psutil
fi

# Get a sample image path
BASE_DIR="/scratch/qgs8612/Experiment"
SAMPLE_IMAGE=$(find "$BASE_DIR" -name "*.tif" -type f | head -1)

if [ -z "$SAMPLE_IMAGE" ]; then
    echo "Error: No sample image found in $BASE_DIR"
    exit 1
fi

echo "Using sample image: $SAMPLE_IMAGE"
echo ""

# Run memory profiler
cd /home/qgs8612/planarian_smFISH
python3 tests/memory_profiler.py \
    --image "$SAMPLE_IMAGE" \
    --tile-size 16 128 128 \
    --n-tiles 20 \
    --test-files 100 \
    --all

echo ""
echo "=========================================="
echo "Memory profiling complete"
echo "=========================================="