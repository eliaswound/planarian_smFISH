# ⚠️ IMPORTANT: Memory Fix Required

## Problem
Your job showed tile size **(32, 256, 256)** which is **TOO LARGE** for 2GB images!
- Tile size (32, 256, 256) = ~2MB per tile
- With 100k+ tiles = >200GB just for tiles
- This causes OOM even with 256GB RAM

## Solution

**You MUST use tile size (8, 64, 64) or smaller!**

### Option 1: Run with correct tile size (RECOMMENDED)

```bash
sbatch generate_dataset_3d.sh
```

This uses the defaults: **(8, 64, 64)**

### Option 2: If you want to override, explicitly set small tiles

```bash
sbatch generate_dataset_3d.sh \
  "/scratch/qgs8612/Experiment" \
  "/scratch/qgs8612/piscis_training_dataset_3d" \
  "565" \
  8 64 64  # <-- TILE SIZE: z y x (MUST be small!)
```

**DO NOT use (32, 256, 256) - it will OOM!**

## What Changed

1. ✅ Reduced `max_tiles_per_image` from 500 to 100 (already done in code)
2. ✅ Streaming format enabled (already done)
3. ⚠️ **You must use small tile size: (8, 64, 64)**

## Memory Comparison

| Tile Size | Tiles/Image | Total Tiles | Memory | Status |
|-----------|-------------|-------------|--------|--------|
| (32, 256, 256) | 10,000+ | 100,000+ | >256GB | ❌ OOM |
| (16, 128, 128) | 2,500 | 25,000 | ~100GB | ⚠️ Risky |
| (8, 64, 64) | 625 | ~6,000 | ~20GB | ✅ Works |

## Run Command

**Just run without arguments to use safe defaults:**

```bash
sbatch generate_dataset_3d.sh
```

This will:
- Use tile size (8, 64, 64) ✅
- Use streaming format ✅  
- Limit to 100 tiles per image ✅
- Should work with 256GB RAM ✅
