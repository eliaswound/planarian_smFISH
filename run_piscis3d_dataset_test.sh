#!/bin/bash

# Simple script to test 3D Piscis dataset loading on CPU (login node)

# Load Python module (no GPU module needed on login node)
module load python-miniconda3

# Force JAX to use CPU and skip CUDA backend
export JAX_PLATFORMS=cpu

echo "Using Python: $(which python)"
python --version
echo "JAX_PLATFORMS: $JAX_PLATFORMS"

# Default parameters (can be overridden via env vars)
MODEL_NAME=${MODEL_NAME:-"spot_detection_3d_v1"}
DATASET_PATH=${DATASET_PATH:-"/scratch/qgs8612/piscis_training_dataset_3d"}
OUTPUT_DIR=${OUTPUT_DIR:-"/scratch/qgs8612/piscis3d_dataset"}

echo "============================================================"
echo "3D Piscis Dataset Loading Test (CPU)"
echo "============================================================"
echo "Model name:   $MODEL_NAME"
echo "Dataset path: $DATASET_PATH"
echo "Output dir:   $OUTPUT_DIR"
echo "============================================================"

cd /home/qgs8612/planarian_smFISH

python tests/train_piscis_3d.py \
  --model_name "$MODEL_NAME" \
  --dataset_path "$DATASET_PATH" \
  --output_dir "$OUTPUT_DIR"

