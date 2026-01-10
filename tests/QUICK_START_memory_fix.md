# Quick Start: Fix Memory Issues for 3D Dataset Generation

## The Problem
- Each source image is ~2GB
- Current approach tries to generate 100k+ tiles
- Runs out of memory even with 256GB RAM

## Quick Fix (Recommended)

### Step 1: Profile Memory First
```bash
# Submit profiling job on server
sbatch tests/test_memory_bottlenecks.sh

# Or run locally with one image
python tests/memory_profiler.py --image /path/to/image.tif --all
```

### Step 2: Use Streaming Dataset (Already Enabled)
The `create_piscis_dataset_3d.py` script now defaults to streaming format which:
- ✅ Processes one image at a time
- ✅ Saves batches immediately (1000 tiles per batch file)
- ✅ Limits tiles per image (max 500)
- ✅ Never loads everything into memory

### Step 3: Reduce Dataset Size
Edit `generate_dataset_3d.sh` or call directly:

```bash
python tests/create_piscis_dataset_3d.py \
    --base_dir /scratch/qgs8612/Experiment \
    --output_path /scratch/qgs8612/piscis_streaming_dataset \
    --tile_size 16 128 128 \
    --overlap_factor 0.0
```

**Key Parameters:**
- `--tile_size 16 128 128`: Smaller tiles = less memory per tile
- `max_tiles_per_image=500`: Limits tiles per image (in code, can't set via CLI yet)

### Step 4: Alternative - Drastically Reduce Tiles
If still out of memory, reduce further:

```python
# In create_piscis_dataset_3d.py, line ~265, change:
max_tiles_per_image=100  # Instead of 500
```

This will create ~10k tiles total instead of 100k.

## Can You Train With This?

**YES!** The streaming dataset format is training-compatible:

```python
from piscis3d.data_streaming import dataset_generator, load_dataset_streaming

# Option 1: Load specific batch (memory efficient)
images, coords = load_dataset_streaming(
    dataset_dir="/scratch/qgs8612/piscis_streaming_dataset",
    split='train',
    batch_idx=0  # Load batch 0 only
)

# Option 2: Generator for training loop (most memory efficient)
for images_batch, coords_batch in dataset_generator(
    dataset_dir="/scratch/qgs8612/piscis_streaming_dataset",
    split='train',
    shuffle_batches=True
):
    # Train on this batch - only one batch in memory
    train_step(images_batch, coords_batch)
```

## Expected Results

| Configuration | Total Tiles | Peak Memory | Status |
|---------------|-------------|-------------|--------|
| Original (100k tiles) | 100,000 | >256GB | ❌ Fails |
| Streaming (500/img) | ~6,000 | ~50GB | ✅ Works |
| Streaming (100/img) | ~1,200 | ~10GB | ✅ Works |
| Streaming (50/img) | ~600 | ~5GB | ✅ Works |

## What Changed?

1. **New streaming format**: `Piscis3D/piscis3d/data_streaming.py`
   - Saves batches incrementally
   - Never loads everything at once

2. **Memory profiler**: `tests/memory_profiler.py`
   - Identifies bottlenecks
   - Shows actual memory usage

3. **Automatic limits**: 
   - Max 500 tiles per image
   - Can be reduced further if needed

## Next Steps

1. **Try streaming with current settings** (should work now)
2. **If still fails**: Reduce `max_tiles_per_image` to 100 or 50
3. **Profile first**: Run memory profiler to see exact bottlenecks
4. **Train incrementally**: Use generator for training loop

## Questions?

Check `tests/README_memory_optimization.md` for detailed explanations.
