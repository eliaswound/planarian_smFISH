#!/bin/bash
echo "=== Checking output logs ==="
ls -lh piscis3d_dataset_generation*.log 2>/dev/null || echo "No log files found"
echo ""
echo "=== Recent output ==="
tail -50 piscis3d_dataset_generation_output.log 2>/dev/null | tail -30
echo ""
echo "=== Recent errors ==="
tail -50 piscis3d_dataset_generation_error.log 2>/dev/null | tail -30
