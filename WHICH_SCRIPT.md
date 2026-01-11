# Which Script to Use: generate_dataset.sh vs generate_dataset_3d.sh

## For 3D Dataset Generation (Current Work)

**Use: `generate_dataset_3d.sh`**

This script:
- Runs `tests/create_piscis_dataset_3d.py`
- Generates **3D** Piscis training datasets
- Uses `Piscis3D/piscis3d/data_streaming.py` for memory-efficient 3D dataset generation
- Creates 3D tiles (z, y, x) from 3D images
- Default tile size: `(8, 64, 64)` - optimized for memory
- Job name: `Piscis3D_Dataset_Generation`

## For 2D Dataset Generation (Original Piscis)

**Use: `generate_dataset.sh`**

This script:
- Runs `tests/create_piscis_dataset.py`
- Generates **2D** Piscis training datasets (original Piscis format)
- Uses `piscis.data.generate_dataset` (original 2D Piscis)
- Converts 3D images to 2D (max projection) and 3D coordinates to 2D
- Job name: `Piscis_Dataset_Generation`

## Summary

Since you're working on **3D training**, you should use:

```bash
sbatch generate_dataset_3d.sh
```

**NOT** `generate_dataset.sh` (that's for 2D, which you're not using anymore).

## Quick Check

To verify you're using the right script, check what Python script it calls:

```bash
# Check 3D script
grep "create_piscis_dataset" generate_dataset_3d.sh

# Should show:
# tests/create_piscis_dataset_3d.py

# Check 2D script  
grep "create_piscis_dataset" generate_dataset.sh

# Should show:
# tests/create_piscis_dataset.py
```
