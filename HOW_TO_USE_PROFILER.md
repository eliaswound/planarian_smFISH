# How to Use the Step-by-Step Memory Profiler

## What It Does

The profiler tracks memory usage at **each step** of dataset generation and logs:
- Memory before/after each step
- Memory increase per step
- **Exact file and line number** where each step runs
- **Which step caused OOM** (if it fails)

## How to Run

Just run your script normally - the profiler is automatically enabled:

```bash
sbatch generate_dataset_3d.sh
```

The profiler will automatically:
1. Start tracking memory
2. Add checkpoints at each major step
3. Log everything to a file

## Where to Find the Log

After running (even if it fails), check:

```bash
# The log is saved in the parent directory of your output
cat memory_profiling.log

# Or find it
find . -name "memory_profiling.log" -type f

# Check the output directory parent
ls -la /scratch/qgs8612/
cat /scratch/qgs8612/memory_profiling.log
```

## What the Log Shows

Example output:
```
======================================================================
STEP-BY-STEP MEMORY PROFILING STARTED
======================================================================
Initial memory: RSS=2.50 GB, Available=240.00 GB
======================================================================

======================================================================
STEP: Step 1: Count valid tiles
Location: generate_dataset_3d_streaming:Step1
Memory before: RSS=2.50 GB
======================================================================

STEP COMPLETE: Step 1: Count valid tiles
Memory after: RSS=2.55 GB (+50.00 MB)
✓ Checkpoint: Step 1: Count valid tiles
  Location: generate_dataset_3d_streaming:Step1
  Memory: RSS=2.55 GB (+50.00 MB, 240.00 GB available)

======================================================================
STEP: Step 2: Shuffle and split
Memory before: RSS=2.55 GB
======================================================================

STEP COMPLETE: Step 2: Shuffle and split
Memory after: RSS=2.60 GB (+50.00 MB)

======================================================================
STEP: Step 3: Process train split
Memory before: RSS=2.60 GB
======================================================================

❌ OUT OF MEMORY in step: Step 3: Process train split
Error: ...
Memory at failure: RSS=256.00 GB
Memory increase in this step: 253.40 GB

Top memory allocations in this step:
  150.00 GB - data_streaming.py:249 (tile extraction)
  100.00 GB - data_streaming.py:272 (batch_image_list.append)
  ...
```

## Interpreting Results

### If OOM Occurs

1. **Find the step that failed** - look for "❌ OUT OF MEMORY"
2. **Check memory before** - see how much was already used
3. **Check the increase** - how much memory that step tried to use
4. **Check the location** - file:line number where it failed
5. **Check top allocations** - see which operations used most memory

### Example Analysis

If you see:
```
STEP: Step 3: Process train split
Memory before: RSS=50.00 GB
❌ OUT OF MEMORY
Memory increase in this step: 200.00 GB
Location: data_streaming.py:272 (batch_image_list.append)
```

This tells you:
- **Step 3** is the problem
- **Line 272** in `data_streaming.py` (accumulating tiles in list)
- The step tried to use **200GB** additional memory
- Likely cause: **Too many tiles in batch_image_list**

**Solution**: Reduce `BATCH_SIZE` or `max_tiles_per_image`

## Common Patterns

| Step That Fails | Likely Cause | Solution |
|-----------------|--------------|----------|
| Step 1 | Too many tiles found | Reduce `max_tiles_per_image` to 25 or 10 |
| Step 2 | Shuffling too many tiles | Unlikely, but reduce tiles per image |
| Step 3 (train) | Batch too large | Reduce `BATCH_SIZE` from 500 to 250 or 100 |
| Step 3 (test) | Same as train | Same solution |
| During batch processing | Accumulating tiles | Reduce `BATCH_SIZE` or process fewer images |

## Next Steps

1. **Run the script** with profiler enabled
2. **Wait for it to fail** (or succeed)
3. **Check `memory_profiling.log`**
4. **Identify the problematic step**
5. **Adjust parameters** based on what you see
6. **Run again** and compare

## Manual Checkpoints

You can also add manual checkpoints in your code:
```python
from step_memory_profiler import get_profiler

profiler = get_profiler()
if profiler:
    profiler.checkpoint("Custom checkpoint name", "file.py:123")
```
