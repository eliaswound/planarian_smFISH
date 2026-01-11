#!/bin/bash
# Quick script to check key info from the log
echo "=== Tile Size ==="
grep "Tile size" piscis3d_dataset_generation_output.log

echo ""
echo "=== Memory Check ==="
grep -A 30 "MEMORY CHECK\|Memory Check\|Memory Status" piscis3d_dataset_generation_output.log | head -40

echo ""
echo "=== Number of Tiles Found ==="
grep -i "found.*tiles\|valid tiles" piscis3d_dataset_generation_output.log

echo ""
echo "=== Last Successful Step ==="
grep -E "Step [0-9]:" piscis3d_dataset_generation_output.log | tail -5

echo ""
echo "=== Errors ==="
grep -i "error\|warning\|oom\|killed" piscis3d_dataset_generation_output.log | tail -10
