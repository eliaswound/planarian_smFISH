# What to Check When Profiler Log Not Found

## Step 1: Check Output Log

The output log will tell you if the profiler started:

```bash
# Check for profiler messages
grep -i "memory profiling\|profiler" piscis3d_dataset_generation_output.log

# Check last 50 lines to see where it failed
tail -50 piscis3d_dataset_generation_output.log

# Check for errors
grep -i "error\|warning" piscis3d_dataset_generation_output.log | tail -20
```

## Step 2: Look for These Messages

### If Profiler Started Successfully:
You should see:
```
======================================================================
Step-by-step memory profiling enabled
Log file: /scratch/qgs8612/memory_profiling.log
======================================================================
```

### If Profiler Failed:
You might see:
```
Warning: Step-by-step memory profiler not available
```
or
```
Warning: Could not initialize memory profiler: [error message]
```

## Step 3: Check Error Log

```bash
# Check error log for import errors
tail -50 piscis3d_dataset_generation_error.log
```

## Step 4: Even Without Profiler

Even if the step profiler didn't work, the **memory monitor** should still work. Look for:

```
MEMORY CHECK AND TROUBLESHOOTING
============================================================
Memory Status: Initial
============================================================
```

This gives you memory information even without the step profiler.

## Most Likely Issue

Since the log file wasn't created, the profiler probably didn't initialize. This could be because:

1. **Import failed**: `step_memory_profiler.py` not in Python path
2. **Dependency missing**: `psutil` not installed
3. **Early failure**: Job failed before profiler could start

**Check the output log first** - it will tell you what happened.
