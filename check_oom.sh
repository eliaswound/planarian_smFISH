#!/bin/bash
# Script to check OOM details from output log

LOG_FILE="piscis3d_dataset_generation_output.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Output log not found: $LOG_FILE"
    exit 1
fi

echo "=== Checking for profiler messages ==="
grep -i "memory profiling\|profiler" "$LOG_FILE" | head -10

echo ""
echo "=== Checking for memory check messages ==="
grep -i "MEMORY CHECK\|Memory Status\|memory status" "$LOG_FILE" | head -10

echo ""
echo "=== Last 50 lines (to see where it failed) ==="
tail -50 "$LOG_FILE"

echo ""
echo "=== Checking for errors ==="
grep -i "error\|warning\|oom\|out of memory" "$LOG_FILE" | tail -20

echo ""
echo "=== Checking tile size information ==="
grep -i "tile size\|Tile size" "$LOG_FILE" | tail -10
