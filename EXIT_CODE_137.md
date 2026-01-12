# Exit Code 137 - Process Killed (OOM)

## What Exit Code 137 Means

Exit code 137 = 128 + 9, where 9 is SIGKILL
- The process was **killed by the operating system**
- Usually caused by **OOM (Out of Memory) killer**
- The job requested 256GB but still ran out of memory

## What to Check

### 1. Check the Error Log

```bash
tail -50 piscis3d_dataset_generation_error.log
```

This will show:
- Any Python errors before the kill
- Traceback if it crashed
- May be empty if it was just killed by OOM

### 2. Check Full Output Log

The output log you showed is very short - it might have more content:

```bash
# Check how many lines
wc -l piscis3d_dataset_generation_output.log

# See last 100 lines
tail -100 piscis3d_dataset_generation_output.log

# Check if it got to dataset loading
grep -i "loading\|scanning\|found.*images" piscis3d_dataset_generation_output.log
```

### 3. Check SLURM Job Status

```bash
sacct -j 5441234 --format=JobID,JobName,State,ExitCode,MaxRSS,MaxVMSize,Elapsed
```

This will show:
- Actual memory used (MaxRSS)
- Whether it hit the memory limit
- How long it ran before being killed

## Likely Causes

Given that it failed very quickly (based on short log), it likely:

1. **Failed during initialization** - Import errors, path issues
2. **Failed immediately when loading data** - Even with streaming, if something loads everything into memory
3. **Python path issues** - Script can't find modules, causing early failure

## Next Steps

1. **Check the error log** to see if there are any errors before the kill
2. **Check full output log** to see how far it got
3. **Check SLURM job details** to see actual memory usage
4. **Look for import errors** - The script might be failing before it even starts

The fact that the log is so short suggests it failed very early, possibly during:
- Module imports
- Path setup
- Initial data loading

Run the check script and share the results!
