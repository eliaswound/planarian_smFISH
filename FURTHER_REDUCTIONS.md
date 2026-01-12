# Further Memory Reductions if Still OOM

If the job is still running out of memory even with tile size (8, 64, 64), here are further reductions:

## Option 1: Reduce max_tiles_per_image

Currently set to 50. Can reduce to 25 or even 10.

**Edit:** `tests/create_piscis_dataset_3d.py`, line ~350

Change:
```python
max_tiles_per_image = 50  # Reduced from 100 - very conservative
```

To:
```python
max_tiles_per_image = 25  # Even more conservative for 2GB images
```

Or:
```python
max_tiles_per_image = 10  # Very aggressive limit
```

## Option 2: Increase min_spots

Filter out more tiles by requiring more spots per tile.

**Edit:** `generate_dataset_3d.sh` or pass as argument:

```bash
sbatch generate_dataset_3d.sh [base_dir] [output] [wavelength] [z] [y] [x] 5
#                                                                        ^
#                                                                   min_spots=5
```

This will only keep tiles with 5+ spots, reducing total tiles.

## Option 3: Process Fewer Images

Instead of processing all 12 images, process in batches:

1. Edit `tests/create_piscis_dataset_3d.py` to add a `max_images` parameter
2. Or manually exclude more conditions

## Option 4: Check What's Actually Happening

Before making more changes, check the output log to see:
- Where exactly it fails
- What tile size was actually used
- What memory was available when it failed

Run:
```bash
tail -100 piscis3d_dataset_generation_output.log
```

## Recommended Next Steps

1. **First, check the output log** to see what happened
2. **Verify tile size** was actually (8, 64, 64)
3. **Check memory status** messages to see available memory
4. **Then reduce max_tiles_per_image** if needed

Don't reduce blindly - check the log first to understand where it failed!
