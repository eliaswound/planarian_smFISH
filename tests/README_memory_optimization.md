# Memory Optimization Guide for 3D Dataset Generation

## Problem
Generating a 3D training dataset from 2GB images with 100k+ tiles causes out-of-memory errors even with 256GB RAM.

## Root Causes Identified

1. **Creating too many individual tile files**: 100k tiles = 200k files (tile + coord), causing filesystem overhead
2. **Loading everything back into memory**: Even with batching, concatenating all tiles at the end loads everything
3. **No limits on tiles per image**: Some images may generate thousands of tiles
4. **Dataset format requires full loading**: The original npz format loads all tiles at once

## Solutions Implemented

### 1. Memory Profiler (`tests/memory_profiler.py`)
Run this first to identify bottlenecks:
```bash
python tests/memory_profiler.py --image /path/to/sample.tif --all
```

### 2. Streaming Dataset Format (`Piscis3D/piscis3d/data_streaming.py`)
- **Key advantage**: Saves tiles in batches (1000 tiles per file) instead of one large file
- **Memory efficient**: Only processes one image at a time, saves batches immediately
- **Training compatible**: Can be loaded incrementally during training
- **Limits tiles per image**: Default max 500 tiles per image to prevent memory spikes

### 3. Recommendations

#### Option A: Use Streaming Format (Recommended)
```python
# In create_piscis_dataset_3d.py, use_streaming=True (default)
generate_piscis_dataset_3d(
    output_path="/scratch/qgs8612/piscis_streaming_dataset",
    tile_size=(16, 128, 128),  # Smaller tiles
    max_tiles_per_image=500    # Limit tiles
)
```

#### Option B: Reduce Dataset Size Drastically
- Reduce `tile_size` to (8, 64, 64) - 8x smaller
- Increase `min_spots` to filter out low-density tiles
- Use `max_tiles_per_image=100` to limit tiles per image
- This will give you ~10k tiles instead of 100k

#### Option C: Process Images in Subsets
Split your 12 images into batches of 3-4 images, generate datasets separately, then combine during training.

## Testing Memory Usage

```bash
# Submit memory profiling job
sbatch tests/test_memory_bottlenecks.sh

# Or run locally with a sample image
python tests/memory_profiler.py \
    --image /path/to/your/image.tif \
    --tile-size 16 128 128 \
    --n-tiles 20 \
    --all
```

## Training with Streaming Dataset

The streaming dataset can be used with a generator:
```python
from piscis3d.data_streaming import dataset_generator

# During training, iterate over batches
for images_batch, coords_batch in dataset_generator(
    dataset_dir="/scratch/qgs8612/piscis_streaming_dataset",
    split='train',
    shuffle_batches=True,
    rng_key=key
):
    # Train on this batch
    # Only one batch in memory at a time
    pass
```

## Expected Memory Usage

| Approach | Peak Memory | Notes |
|----------|-------------|-------|
| Original (100k tiles) | >256GB | Fails |
| Streaming (limited tiles) | ~50GB | Works with 256GB |
| Streaming (500 tiles/img) | ~20GB | Recommended |
| Small dataset (10k tiles) | ~10GB | Fastest |

## Next Steps

1. **Profile first**: Run memory profiler to see actual bottlenecks
2. **Use streaming format**: Generate dataset with `use_streaming=True`
3. **Limit tiles**: Set `max_tiles_per_image=500` or lower
4. **Smaller tiles**: Use `tile_size=(16, 128, 128)` or smaller
5. **Test training**: Verify you can load and train with the streaming dataset
