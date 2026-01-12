#!/bin/bash
# Script to check which log file corresponds to which job

echo "=== Checking log files ==="
echo ""
echo "3D logs (from generate_dataset_3d.sh):"
ls -lh piscis3d_dataset_generation*.log 2>/dev/null | head -5

echo ""
echo "2D logs (from generate_dataset.sh):"
ls -lh piscis_dataset_generation*.log 2>/dev/null | head -5

echo ""
echo "=== Latest 3D output log (if exists) ==="
if [ -f "piscis3d_dataset_generation_output.log" ]; then
    echo "Last 30 lines:"
    tail -30 piscis3d_dataset_generation_output.log
else
    echo "3D output log not found"
fi

echo ""
echo "=== Check job names in sacct ==="
echo "Run: sacct -X | grep Piscis"
