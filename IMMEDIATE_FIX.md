# Immediate Fix for OOM Issue

## Problem Summary

- Job ran for 7 minutes before OOM kill
- Used exactly 256GB (hit memory limit)
- Only 23 lines in output log
- Failed before "Loading dataset paths only" message
- MaxRSS: 268407260K ≈ 256GB

## Likely Cause

The script is failing **very early**, possibly during:
1. Python imports (JAX, NumPy, tifffile, etc.)
2. Module initialization
3. PYTHONPATH setup

But it ran for **7 minutes**, which suggests something is loading/initializing that consumes memory.

## Quick Check

First, check the **full output log** to see exactly where it stopped:

```bash
cat piscis3d_dataset_generation_output.log
```

This will show you the last message before it died.

## Potential Issues

### 1. PYTHONPATH Issue

The script adds to PYTHONPATH:
```bash
export PYTHONPATH="$HOME/planarian_smFISH/Piscis3D:$HOME/planarian_smFISH/tests:$PYTHONPATH"
```

If `Piscis3D` directory contains large files or circular imports, this could cause issues.

### 2. JAX Initialization

JAX can consume significant memory on initialization, especially if CUDA libraries are loaded.

### 3. Import Order

Some imports might trigger loading of data or initialization of large objects.

## Next Steps

1. **Check full log** to see exact failure point
2. **Test imports separately** to see which one fails
3. **Reduce PYTHONPATH** to minimal required paths
4. **Try running with fewer imports** to isolate the issue

But first, **check the full output log** to see what the last message was!
