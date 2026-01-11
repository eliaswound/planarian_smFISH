# How to Find the Memory Profiling Log

## Where is it saved?

The memory profiling log is saved in the **parent directory** of your output path.

For example:
- If output path is: `/scratch/qgs8612/piscis_training_dataset_3d`
- Log file will be: `/scratch/qgs8612/memory_profiling.log`

## How to Find It

### Option 1: Check the output log

Look for profiler initialization messages:
```bash
grep -i "memory profiling\|profiler" piscis3d_dataset_generation_output.log
```

This will show you if the profiler started and where the log is saved.

### Option 2: Check common locations

```bash
# In the output directory's parent
cat /scratch/qgs8612/memory_profiling.log

# Or find it
find /scratch/qgs8612 -maxdepth 2 -name "memory_profiling.log" -type f

# Or in current directory
find . -name "memory_profiling.log" -type f
```

### Option 3: Use the helper script

```bash
bash check_profiler_output.sh
```

## Why Might It Not Exist?

1. **Profiler failed to initialize**: Check output log for import errors
2. **Script failed before profiler started**: Check for early errors
3. **Wrong path**: The log might be in a different location

## What to Check in Output Log

Look for these messages:
```
Step-by-step memory profiling enabled
Log file: /path/to/memory_profiling.log
```

If you see:
```
Warning: Step-by-step memory profiler not available
```
Then the profiler didn't initialize, and no log will be created.

## If Profiler Didn't Initialize

Check the output log for:
- Import errors
- Path issues
- Python path problems

The profiler needs `step_memory_profiler.py` to be in the Python path. It should be in the `tests/` directory.
