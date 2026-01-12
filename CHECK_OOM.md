# How to Check OOM (Out of Memory) Issues

## Quick Check Commands

Run these on your server to diagnose the OOM:

```bash
# 1. Check if profiler initialized
grep -i "memory profiling\|profiler" piscis3d_dataset_generation_output.log

# 2. Check memory status messages
grep -i "MEMORY CHECK\|Memory Status" piscis3d_dataset_generation_output.log

# 3. See last 50 lines (where it failed)
tail -50 piscis3d_dataset_generation_output.log

# 4. Check for tile size info
grep -i "tile size" piscis3d_dataset_generation_output.log | tail -10

# 5. Check for errors
grep -i "error\|warning\|oom" piscis3d_dataset_generation_output.log | tail -20
```

## What to Look For

### If Profiler Initialized:
You should see:
```
======================================================================
Step-by-step memory profiling enabled
Log file: /scratch/qgs8612/memory_profiling.log
======================================================================
```

If you don't see this, the profiler didn't initialize (likely failed before it could start).

### Memory Check Messages:
Even without profiler, you should see:
```
MEMORY CHECK AND TROUBLESHOOTING
============================================================
Memory Status: Initial
```

### Last Lines Before OOM:
Check what the last operation was before OOM. Common patterns:
- "Scanning image X/12" → Failed during image scanning
- "Processing train split" → Failed during dataset generation
- "Loading dataset paths" → Failed early

### Tile Size:
Make sure you see:
```
Tile size (z, y, x): (8, 64, 64)
```

If you see a larger tile size, that's the problem!

## If Still OOM with (8, 64, 64)

If tile size is correct but still OOM, try:

1. **Reduce max_tiles_per_image** (currently 50)
   - Edit `tests/create_piscis_dataset_3d.py`, line ~350
   - Change `max_tiles_per_image = 50` to `max_tiles_per_image = 25`

2. **Check available memory**
   ```bash
   # On login node
   free -h
   ```

3. **Reduce number of images processed**
   - Process fewer images at a time
   - Or increase `min_spots` to filter out more tiles

4. **Check for memory leaks**
   - Look for patterns in output log
   - Check if memory increases gradually or spikes suddenly

## Next Steps

After checking the log, we can:
1. Identify the exact step causing OOM
2. Reduce parameters further if needed
3. Process images in smaller batches
