#!/bin/bash
# Quick test script - choose what to run

echo "=========================================="
echo "3D Dataset Generation Test Options"
echo "=========================================="
echo ""
echo "1. Run dataset generation (recommended):"
echo "   sbatch generate_dataset_3d.sh"
echo ""
echo "2. Profile memory first (optional):"
echo "   sbatch tests/test_memory_bottlenecks.sh"
echo ""
echo "3. Run directly (on login node, for testing):"
echo "   python tests/create_piscis_dataset_3d.py --base_dir /scratch/qgs8612/Experiment --output_path /scratch/qgs8612/piscis_streaming_dataset_test"
echo ""
echo "=========================================="
echo ""
read -p "Which option? (1/2/3) " choice

case $choice in
    1)
        echo "Submitting dataset generation job..."
        sbatch generate_dataset_3d.sh
        echo "Job submitted! Check status with: squeue -u \$USER"
        echo "Monitor output: tail -f piscis3d_dataset_generation_output.log"
        ;;
    2)
        echo "Submitting memory profiling job..."
        sbatch tests/test_memory_bottlenecks.sh
        echo "Job submitted! Check status with: squeue -u \$USER"
        echo "Monitor output: tail -f memory_profile_*.log"
        ;;
    3)
        echo "Running directly (make sure you're on login node with conda activated)..."
        eval "$(conda shell.bash hook)"
        conda activate smfish_env
        export JAX_PLATFORMS=cpu
        export PYTHONPATH="$HOME/planarian_smFISH/Piscis3D:$PYTHONPATH"
        python tests/create_piscis_dataset_3d.py \
            --base_dir /scratch/qgs8612/Experiment \
            --output_path /scratch/qgs8612/piscis_streaming_dataset_test \
            --tile_size 16 128 128 \
            --overlap_factor 0.0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac
