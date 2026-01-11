# Troubleshooting Guide for OOM Errors

## Quick Diagnosis

If you get an OOM (Out of Memory) error, check these:

### 1. Verify Tile Size

**CRITICAL**: The tile size MUST be small for 2GB images!

```bash
# Check what tile size is being used
grep "Tile size" piscis3d_dataset_generation_output.log
```

**Should show**: `Tile size (z, y, x): (8, 64, 64)`  
**If it shows**: `(32, 256, 256)` or larger → **THIS IS THE PROBLEM!**

### 2. Check Memory Before Running

```bash
# On login node, check available memory
free -h

# Or run memory check
python tests/memory_monitor.py --check --tile-size 8 64 64 --n-images 12
```

### 3. Verify Script Version

Make sure you're using the updated script:

```bash
# Check tile size defaults in script
grep "TILE_SIZE" generate_dataset_3d.sh | head -3
```

Should show:
```
TILE_SIZE_Z=${4:-8}
TILE_SIZE_Y=${5:-64}
TILE_SIZE_X=${6:-64}
```

## Common Issues

### Issue 1: Tile Size Too Large

**Symptom**: Output shows `Tile size (z, y, x): (32, 256, 256)`

**Cause**: Script was run with arguments that override defaults

**Fix**: Run without arguments:
```bash
sbatch generate_dataset_3d.sh
```

Or explicitly set small tiles:
```bash
sbatch generate_dataset_3d.sh \
  "/scratch/qgs8612/Experiment" \
  "/scratch/qgs8612/piscis_training_dataset_3d" \
  "565" \
  8 64 64  # <-- MUST be small!
```

### Issue 2: Too Many Tiles Per Image

**Symptom**: OOM even with small tile size

**Fix**: The code now automatically reduces `max_tiles_per_image` based on tile size, but you can also manually reduce it in `create_piscis_dataset_3d.py` line 265:
```python
max_tiles_per_image=50  # Reduce from 100
```

### Issue 3: Memory Leak

**Symptom**: Memory usage keeps growing during processing

**Fix**: The streaming format should prevent this, but if it happens:
1. Check for memory leaks in the log
2. Reduce batch size further
3. Process images in smaller groups

## Memory Monitoring

### During Job Execution

Monitor memory in real-time:
```bash
# On compute node (if you have access)
watch -n 5 'free -h && ps aux | grep python | head -5'
```

### After OOM Error

Check the error log:
```bash
cat piscis3d_dataset_generation_error.log
```

Look for:
- Memory check output
- Tile size being used
- Number of tiles being processed
- Where it failed (which step)

## Step-by-Step Debugging

1. **Check tile size in output log**:
   ```bash
   grep "Tile size" piscis3d_dataset_generation_output.log
   ```

2. **Check memory check results**:
   ```bash
   grep -A 20 "MEMORY CHECK" piscis3d_dataset_generation_output.log
   ```

3. **Check how many tiles were found**:
   ```bash
   grep "Found.*valid tiles" piscis3d_dataset_generation_output.log
   ```

4. **Check which step failed**:
   ```bash
   grep -E "Step [0-9]:" piscis3d_dataset_generation_output.log | tail -5
   ```

## Solutions by Symptom

| Symptom | Likely Cause | Solution |
|---------|--------------|----------|
| OOM immediately | Tile size too large | Use `--tile_size 8 64 64` |
| OOM during Step 1 | Too many tiles found | Increase `min_spots` or reduce `max_tiles_per_image` |
| OOM during Step 3 | Batch too large | Already using batch_size=1, check tile size |
| OOM during Step 4 | Too many tiles in memory | Reduce `max_tiles_per_image` to 50 or 25 |

## Emergency Fixes

If nothing works, try these extreme reductions:

```bash
# Ultra-small tiles
sbatch generate_dataset_3d.sh \
  "/scratch/qgs8612/Experiment" \
  "/scratch/qgs8612/piscis_training_dataset_3d" \
  "565" \
  4 32 32  # Even smaller!

# Or edit create_piscis_dataset_3d.py line 265:
max_tiles_per_image=25  # Very aggressive limit
```

## Getting Help

If you're still stuck, provide:
1. Output log showing tile size
2. Memory check output
3. Error log
4. Output of: `free -h` on the compute node
