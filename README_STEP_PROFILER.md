# Step-by-Step Memory Profiler

## What It Does

The step profiler tracks memory usage at each step of dataset generation to identify exactly where OOM occurs.

## How It Works

1. **Monitors memory before/after each step**
2. **Tracks memory growth between steps**
3. **Records file/line location of each step**
4. **Logs all information to a file**

## Output

After running, check the memory profiling log:

```bash
# The log is saved in the parent directory of your output
cat memory_profiling.log
```

Or find it:
```bash
# Look for memory_profiling.log in the output directory's parent
find . -name "memory_profiling.log" -type f
```

## What to Look For

The log will show:
1. **Memory at start** - baseline
2. **Memory after each step** - see which step causes increase
3. **Memory increase per step** - identify the problematic step
4. **Location of each step** - file and line number

Example output:
```
STEP: Step 1: Count valid tiles
Location: generate_dataset_3d_streaming:Step1
Memory before: RSS=2.50 GB
Memory after: RSS=2.55 GB (+50.00 MB)
STEP COMPLETE: Step 1: Count valid tiles

STEP: Step 2: Shuffle and split
Memory before: RSS=2.55 GB
Memory after: RSS=3.00 GB (+450.00 MB)
⚠️  WARNING: Step caused 450.00 MB memory increase!

STEP: Step 3: Process train split
Memory before: RSS=3.00 GB
❌ OUT OF MEMORY in step: Step 3: Process train split
Memory at failure: RSS=256.00 GB
Memory increase in this step: 253.00 GB
```

## Identifying the Problem

1. **Check which step failed** - look for "OUT OF MEMORY"
2. **See memory before that step** - how much was already used
3. **See the increase** - how much memory that step tried to use
4. **Check the location** - file and line number where it failed

## Common Patterns

- **Step 1 fails**: Too many tiles being counted (reduce max_tiles_per_image)
- **Step 2 fails**: Shuffling large lists (already optimized, shouldn't fail)
- **Step 3 fails**: Processing batches (reduce BATCH_SIZE or max_tiles_per_image)
- **During batch processing**: Accumulating too many tiles (reduce BATCH_SIZE)

## Next Steps After Identifying Problem

Once you know which step fails:

1. **If Step 1 fails**: Reduce `max_tiles_per_image` to 25 or 10
2. **If Step 3 fails**: 
   - Reduce `BATCH_SIZE` from 500 to 250 or 100
   - Reduce `max_tiles_per_image` further
   - Check if specific batch or image causes issue
