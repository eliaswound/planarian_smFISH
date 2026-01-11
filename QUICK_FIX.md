# Quick Fix for Ongoing OOM Issues

## Status
✅ Tile size is correct: (8, 64, 64)  
❌ Still getting OOM errors

## What Changed
1. Reduced `max_tiles_per_image` from 100 → **50** (more aggressive)
2. Reduced batch size from 1000 → **500** tiles per batch file
3. Added memory monitoring and diagnostics

## Next Steps

### 1. Check Memory Usage Details
Run this on your server to see what's in the log:
```bash
grep -A 30 "MEMORY CHECK\|Memory Status" piscis3d_dataset_generation_output.log
```

### 2. If Still OOM, Try Even More Aggressive Limits

Edit `tests/create_piscis_dataset_3d.py` line ~268, change:
```python
max_tiles_per_image = 25  # Even more aggressive
```

Or edit `Piscis3D/piscis3d/data_streaming.py` line ~126, change:
```python
BATCH_SIZE = 250  # Smaller batches
```

### 3. Alternative: Process Fewer Images

If you have 12 images, try processing 6 at a time:

```bash
# First batch: Images 1-6
# (Temporarily move or rename images 7-12)

# Then process second batch
```

### 4. Check Actual Memory Usage

On the compute node (if you can access it during/after job):
```bash
# Check peak memory usage
sacct -j JOBID --format=JobID,MaxRSS,Elapsed

# Or check memory from the log
grep -i "memory\|rss" piscis3d_dataset_generation_output.log
```

## Most Likely Cause

Even with correct tile size, if images are 2GB each:
- Memory-mapping overhead per image
- Accumulating tiles in batches
- System overhead

The new limits (50 tiles/image, 500 tiles/batch) should help, but may need to go lower.

## Recommendation

Try running again with the new limits. If it still fails:
1. Check the memory monitoring output in the log
2. Reduce `max_tiles_per_image` to 25
3. Consider processing images in smaller batches
