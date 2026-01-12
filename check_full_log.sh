#!/bin/bash
# Check full output log to see what happened

LOG_FILE="piscis3d_dataset_generation_output.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "Log file not found: $LOG_FILE"
    exit 1
fi

echo "=== Full output log ==="
cat "$LOG_FILE"

echo ""
echo "=== Checking for specific messages ==="
echo "Looking for 'Starting 3D':"
grep -n "Starting 3D" "$LOG_FILE"

echo ""
echo "Looking for 'Loading dataset':"
grep -n "Loading dataset" "$LOG_FILE"

echo ""
echo "Looking for import errors:"
grep -i "error\|import\|module" "$LOG_FILE"
