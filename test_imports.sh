#!/bin/bash
# Test imports separately to find what's causing OOM

echo "=== Testing imports step by step ==="

PYTHON="/software/2025/jax-fem/0.0.8/bin/python"
export JAX_PLATFORMS=cpu
export PYTHONPATH="$HOME/planarian_smFISH/Piscis3D:$HOME/planarian_smFISH/tests:$PYTHONPATH"

cd $HOME/planarian_smFISH

echo ""
echo "1. Testing basic Python imports..."
$PYTHON -c "import argparse, os, sys, pathlib; print('✓ Basic imports OK')" || echo "✗ Basic imports failed"

echo ""
echo "2. Testing numpy..."
$PYTHON -c "import numpy as np; print('✓ NumPy OK')" || echo "✗ NumPy failed"

echo ""
echo "3. Testing JAX..."
$PYTHON -c "import jax; print('✓ JAX OK')" || echo "✗ JAX failed"

echo ""
echo "4. Testing Piscis3D imports..."
$PYTHON -c "
import sys
from pathlib import Path
piscis3d_path = Path('$HOME/planarian_smFISH/Piscis3D')
sys.path.insert(0, str(piscis3d_path))
try:
    from piscis3d.data_streaming import generate_dataset_3d_streaming
    print('✓ Piscis3D data_streaming import OK')
except Exception as e:
    print(f'✗ Piscis3D import failed: {e}')
" || echo "✗ Piscis3D import failed"

echo ""
echo "5. Testing dataset_loading import..."
$PYTHON -c "
import sys
sys.path.insert(0, '$HOME/planarian_smFISH/tests')
try:
    from dataset_loading import load_dataset_paths_only
    print('✓ dataset_loading import OK')
except Exception as e:
    print(f'✗ dataset_loading import failed: {e}')
" || echo "✗ dataset_loading import failed"

echo ""
echo "6. Testing script import (without running main)..."
$PYTHON -c "
import sys
sys.path.insert(0, '$HOME/planarian_smFISH/tests')
try:
    import create_piscis_dataset_3d
    print('✓ Script import OK')
except Exception as e:
    print(f'✗ Script import failed: {e}')
    import traceback
    traceback.print_exc()
" || echo "✗ Script import failed"

echo ""
echo "=== Import test complete ==="
