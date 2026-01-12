#!/bin/bash
# Script to check 3D job failure details

LOG_FILE="piscis3d_dataset_generation_output.log"
ERR_FILE="piscis3d_dataset_generation_error.log"

echo "=== Checking error log ==="
if [ -f "$ERR_FILE" ]; then
    echo "Last 50 lines of error log:"
    tail -50 "$ERR_FILE"
else
    echo "Error log not found: $ERR_FILE"
fi

echo ""
echo "=== Checking for profiler messages in output ==="
if [ -f "$LOG_FILE" ]; then
    grep -i "memory profiling\|profiler" "$LOG_FILE"
    if [ $? -ne 0 ]; then
        echo "No profiler messages found"
    fi
fi

echo ""
echo "=== Checking for dataset loading messages ==="
if [ -f "$LOG_FILE" ]; then
    grep -i "loading dataset\|found.*images\|scanning" "$LOG_FILE" | tail -20
fi

echo ""
echo "=== Full output log length ==="
if [ -f "$LOG_FILE" ]; then
    wc -l "$LOG_FILE"
    echo "Last 100 lines:"
    tail -100 "$LOG_FILE"
fi

echo ""
echo "=== Checking for memory check messages ==="
if [ -f "$LOG_FILE" ]; then
    grep -i "memory check\|memory status" "$LOG_FILE"
fi
