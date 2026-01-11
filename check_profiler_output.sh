#!/bin/bash
# Script to check for memory profiling output

echo "=== Looking for memory profiling logs ==="
echo ""
echo "1. In current directory:"
find . -maxdepth 1 -name "memory_profiling.log" -type f 2>/dev/null
echo ""

echo "2. In output directory parent:"
OUTPUT_DIR=$(grep "OUTPUT_PATH" generate_dataset_3d.sh 2>/dev/null | head -1 | cut -d'=' -f2 | tr -d '"')
if [ -n "$OUTPUT_DIR" ]; then
    OUTPUT_PARENT=$(dirname "$OUTPUT_DIR")
    echo "Checking: $OUTPUT_PARENT"
    find "$OUTPUT_PARENT" -maxdepth 1 -name "memory_profiling.log" -type f 2>/dev/null
fi
echo ""

echo "3. In scratch directory:"
find /scratch/qgs8612 -maxdepth 2 -name "memory_profiling.log" -type f 2>/dev/null 2>/dev/null | head -5
echo ""

echo "=== Checking for profiler messages in output log ==="
if [ -f "piscis3d_dataset_generation_output.log" ]; then
    echo "Looking for profiler initialization..."
    grep -i "memory profiling\|profiler" piscis3d_dataset_generation_output.log | head -10
else
    echo "Output log not found"
fi
echo ""

echo "=== Checking Python path in script ==="
grep "PYTHONPATH" generate_dataset_3d.sh | grep -v "^#"
