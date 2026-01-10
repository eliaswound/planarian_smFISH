#!/usr/bin/env python3
"""
Memory profiler for 3D dataset generation.
Helps identify memory bottlenecks in the dataset generation process.
"""

import tracemalloc
import psutil
import os
import sys
import gc
from pathlib import Path
import numpy as np
from tifffile import TiffFile, imread, memmap
import linecache

def format_size(bytes_size):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def display_top(snapshot, key_type='lineno', limit=10):
    """Display top memory allocations"""
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print(f"\n{'='*60}")
    print(f"Top {limit} memory allocations:")
    print(f"{'='*60}")
    
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        print(f"#{index}: {format_size(stat.size)} - {frame.filename}:{frame.lineno} - {stat.traceback_format()[-1]}")
        # Show more context
        print(f"    Code: {linecache.getline(frame.filename, frame.lineno).strip()}")
        print()

def check_memory_usage():
    """Check current memory usage"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return {
        'rss': mem_info.rss,  # Resident Set Size (actual RAM used)
        'vms': mem_info.vms,  # Virtual Memory Size
        'percent': process.memory_percent()
    }

def profile_image_loading(image_path, use_memmap=True):
    """Profile memory usage when loading an image"""
    print(f"\n{'='*60}")
    print(f"Profiling image loading: {Path(image_path).name}")
    print(f"{'='*60}")
    
    gc.collect()
    mem_before = check_memory_usage()
    print(f"Memory before: RSS={format_size(mem_before['rss'])}, VMS={format_size(mem_before['vms'])}, {mem_before['percent']:.2f}%")
    
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    try:
        if use_memmap:
            print(f"\nUsing memory mapping...")
            mmap_array = memmap(image_path)
            snapshot2 = tracemalloc.take_snapshot()
            mem_after_mmap = check_memory_usage()
            print(f"Memory after memmap: RSS={format_size(mem_after_mmap['rss'])}, VMS={format_size(mem_after_mmap['vms'])}, {mem_after_mmap['percent']:.2f}%")
            
            # Get image shape
            with TiffFile(image_path) as tif:
                shape = tif.series[0].shape
            print(f"Image shape: {shape}")
            
            # Test reading a small slice
            print(f"\nReading a small slice (first 10x100x100)...")
            slice_data = np.ascontiguousarray(mmap_array[0:10, 0:100, 0:100])
            snapshot3 = tracemalloc.take_snapshot()
            mem_after_slice = check_memory_usage()
            print(f"Memory after slice read: RSS={format_size(mem_after_slice['rss'])}, VMS={format_size(mem_after_slice['vms'])}, {mem_after_slice['percent']:.2f}%")
            print(f"Slice size: {format_size(slice_data.nbytes)}")
            
            top_stats = snapshot3.compare_to(snapshot1, 'lineno')
            print(f"\nMemory growth during slice read:")
            for stat in top_stats[:5]:
                frame = stat.traceback[0]
                print(f"  +{format_size(stat.size)} - {frame.filename}:{frame.lineno}")
            
            del slice_data
            del mmap_array
            
        else:
            print(f"\nUsing imread (full load)...")
            image = imread(image_path)
            snapshot2 = tracemalloc.take_snapshot()
            mem_after_load = check_memory_usage()
            print(f"Memory after imread: RSS={format_size(mem_after_load['rss'])}, VMS={format_size(mem_after_load['vms'])}, {mem_after_load['percent']:.2f}%")
            print(f"Image shape: {image.shape}, size: {format_size(image.nbytes)}")
            
            del image
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        gc.collect()
        tracemalloc.stop()
        mem_final = check_memory_usage()
        print(f"\nMemory after cleanup: RSS={format_size(mem_final['rss'])}, VMS={format_size(mem_final['vms'])}, {mem_final['percent']:.2f}%")
        print(f"Net memory increase: RSS={format_size(mem_final['rss'] - mem_before['rss'])}")

def profile_tile_extraction(image_path, tile_size=(16, 128, 128), n_tiles=10):
    """Profile memory usage when extracting multiple tiles"""
    print(f"\n{'='*60}")
    print(f"Profiling tile extraction: {n_tiles} tiles of size {tile_size}")
    print(f"{'='*60}")
    
    gc.collect()
    mem_before = check_memory_usage()
    print(f"Memory before: RSS={format_size(mem_before['rss'])}")
    
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    try:
        # Open with memmap
        mmap_array = memmap(image_path)
        with TiffFile(image_path) as tif:
            image_shape = tif.series[0].shape
        
        print(f"Image shape: {image_shape}")
        print(f"Tile size: {tile_size}")
        
        # Extract tiles
        tiles = []
        for i in range(min(n_tiles, 10)):  # Limit to 10 for profiling
            z_start = (i * tile_size[0]) % (image_shape[0] - tile_size[0] + 1)
            y_start = (i * tile_size[1]) % (image_shape[1] - tile_size[1] + 1)
            x_start = (i * tile_size[2]) % (image_shape[2] - tile_size[2] + 1)
            
            z_end = z_start + tile_size[0]
            y_end = y_start + tile_size[1]
            x_end = x_start + tile_size[2]
            
            tile = np.ascontiguousarray(mmap_array[z_start:z_end, y_start:y_end, x_start:x_end]).astype(np.float32)
            tiles.append(tile)
            
            if i == 0:
                snapshot2 = tracemalloc.take_snapshot()
                mem_after_first = check_memory_usage()
                print(f"\nAfter 1 tile: RSS={format_size(mem_after_first['rss'])} (+{format_size(mem_after_first['rss'] - mem_before['rss'])})")
            
            if i == 4:
                snapshot3 = tracemalloc.take_snapshot()
                mem_after_five = check_memory_usage()
                print(f"After 5 tiles: RSS={format_size(mem_after_five['rss'])} (+{format_size(mem_after_five['rss'] - mem_before['rss'])})")
        
        snapshot4 = tracemalloc.take_snapshot()
        mem_after_all = check_memory_usage()
        total_tile_size = sum(t.nbytes for t in tiles)
        print(f"\nAfter {len(tiles)} tiles: RSS={format_size(mem_after_all['rss'])} (+{format_size(mem_after_all['rss'] - mem_before['rss'])})")
        print(f"Total tile data size: {format_size(total_tile_size)}")
        print(f"Memory overhead: {format_size(mem_after_all['rss'] - mem_before['rss'] - total_tile_size)}")
        
        # Show top allocations
        top_stats = snapshot4.compare_to(snapshot1, 'lineno')
        print(f"\nTop memory allocations:")
        for stat in top_stats[:10]:
            frame = stat.traceback[0]
            print(f"  {format_size(stat.size)} - {frame.filename}:{frame.lineno}")
        
        # Cleanup
        del tiles
        del mmap_array
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        gc.collect()
        tracemalloc.stop()
        mem_final = check_memory_usage()
        print(f"\nAfter cleanup: RSS={format_size(mem_final['rss'])}")
        print(f"Net memory increase: {format_size(mem_final['rss'] - mem_before['rss'])}")

def profile_file_operations(n_files=1000):
    """Profile memory usage when creating many small files"""
    print(f"\n{'='*60}")
    print(f"Profiling file creation: {n_files} small files")
    print(f"{'='*60}")
    
    import tempfile
    import shutil
    
    gc.collect()
    mem_before = check_memory_usage()
    print(f"Memory before: RSS={format_size(mem_before['rss'])}")
    
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    tmpdir = tempfile.mkdtemp()
    print(f"Temp directory: {tmpdir}")
    
    try:
        tile_size = (16, 128, 128)
        tile_data = np.random.rand(*tile_size).astype(np.float32)
        tile_size_bytes = tile_data.nbytes
        
        file_paths = []
        for i in range(min(n_files, 1000)):  # Limit for profiling
            file_path = os.path.join(tmpdir, f'tile_{i}.npy')
            np.save(file_path, tile_data)
            file_paths.append(file_path)
            
            if i == 0:
                mem_after_first = check_memory_usage()
                print(f"\nAfter 1 file: RSS={format_size(mem_after_first['rss'])}")
            if i == 99:
                mem_after_hundred = check_memory_usage()
                print(f"After 100 files: RSS={format_size(mem_after_hundred['rss'])}")
        
        snapshot2 = tracemalloc.take_snapshot()
        mem_after_all = check_memory_usage()
        print(f"\nAfter {len(file_paths)} files: RSS={format_size(mem_after_all['rss'])}")
        print(f"Total file data size: {format_size(tile_size_bytes * len(file_paths))}")
        
        # Check disk usage
        total_size = sum(os.path.getsize(f) for f in file_paths)
        print(f"Total disk usage: {format_size(total_size)}")
        
        # Try loading files back
        print(f"\nLoading files back...")
        loaded_tiles = []
        for i, file_path in enumerate(file_paths[:10]):  # Load only 10 for profiling
            tile = np.load(file_path)
            loaded_tiles.append(tile)
        
        mem_after_load = check_memory_usage()
        print(f"After loading 10 files: RSS={format_size(mem_after_load['rss'])}")
        
        del loaded_tiles
        del file_paths
        
    finally:
        shutil.rmtree(tmpdir)
        gc.collect()
        tracemalloc.stop()
        mem_final = check_memory_usage()
        print(f"\nAfter cleanup: RSS={format_size(mem_final['rss'])}")
        print(f"Net memory increase: {format_size(mem_final['rss'] - mem_before['rss'])}")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Profile memory usage for dataset generation')
    parser.add_argument('--image', type=str, help='Path to a test image file')
    parser.add_argument('--tile-size', type=int, nargs=3, default=[16, 128, 128], 
                        help='Tile size (z y x)')
    parser.add_argument('--n-tiles', type=int, default=10, help='Number of tiles to extract')
    parser.add_argument('--test-files', type=int, default=100, help='Number of files to test')
    parser.add_argument('--all', action='store_true', help='Run all tests')
    
    args = parser.parse_args()
    
    print(f"{'='*60}")
    print("Memory Profiler for 3D Dataset Generation")
    print(f"{'='*60}")
    print(f"System RAM: {format_size(psutil.virtual_memory().total)}")
    print(f"Available RAM: {format_size(psutil.virtual_memory().available)}")
    print(f"Memory usage: {psutil.virtual_memory().percent:.2f}%")
    
    if args.all or args.image:
        if args.image and os.path.exists(args.image):
            profile_image_loading(args.image, use_memmap=True)
            profile_image_loading(args.image, use_memmap=False)
            profile_tile_extraction(args.image, tuple(args.tile_size), args.n_tiles)
        else:
            print("\nError: Image file not found or not specified")
    
    if args.all:
        profile_file_operations(args.test_files)
    
    print(f"\n{'='*60}")
    print("Profiling complete")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()