# Which Log File to Check

## For 3D Dataset Generation

If you ran `sbatch generate_dataset_3d.sh`, check:

```bash
# Output log (where progress messages go)
tail -100 piscis3d_dataset_generation_output.log

# Error log (where errors go)
tail -50 piscis3d_dataset_generation_error.log
```

**Key differences:**
- Log file name: `piscis3d_dataset_generation_output.log` (has "3d" in name)
- Job name in sacct: `Piscis3D_+` (has "3D")
- Output path: `/scratch/qgs8612/piscis_training_dataset_3d` (has "_3d")

## For 2D Dataset Generation

If you ran `sbatch generate_dataset.sh`, check:

```bash
# Output log
tail -100 piscis_dataset_generation_output.log

# Error log  
tail -50 piscis_dataset_generation_error.log
```

**Key differences:**
- Log file name: `piscis_dataset_generation_output.log` (NO "3d")
- Job name in sacct: `Piscis_+` (NO "3D")
- Output path: `/scratch/qgs8612/piscis_training_dataset` (NO "_3d")

## What You Just Showed Me

The output you showed has:
- Tile size: `256x256` (2D)
- Output path: `/scratch/qgs8612/piscis_training_dataset` (NO "_3d")
- Calling: `piscis.data.generate_dataset()` (2D function)

This is from the **2D script** (`generate_dataset.sh`), NOT the 3D script!

## To Check Your 3D Job

Run:

```bash
# Check the 3D output log
tail -100 piscis3d_dataset_generation_output.log

# Or check which job is which
ls -lht *.log | head -10
```

The 3D log should show:
- "Starting 3D Piscis Dataset Generation"
- "Tile size (z, y, x): (8, 64, 64)"
- Output path: `/scratch/qgs8612/piscis_training_dataset_3d`
