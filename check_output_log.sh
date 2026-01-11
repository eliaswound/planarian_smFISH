#!/bin/bash
# Script to check output log for profiler messages and errors

LOG_FILE="piscis3d_dataset_generation_output.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Output log not found: $LOG_FILE"
    exit 1
fi

echo "=== Checking for profiler messages ==="
grep -i "memory profiling\|profiler" "$LOG_FILE" | head -10

echo ""
echo "=== Checking for errors ==="
grep -i "error\|warning.*profiler\|import.*profiler" "$LOG_FILE" | head -10

echo ""
echo "=== Checking where the script got to ==="
echo "Last 20 lines of output:"
tail -20 "$LOG_FILE"

echo ""
echo "=== Checking for step messages ==="
grep -E "Step [0-9]:" "$LOG_FILE" | tail -5

echo ""
echo "=== Checking for memory check messages ==="
grep -i "memory check\|memory status" "$LOG_FILE" | head -5
