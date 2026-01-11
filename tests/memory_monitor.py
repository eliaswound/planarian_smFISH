#!/usr/bin/env python3
"""
Memory monitoring and troubleshooting utilities for dataset generation.
"""

import psutil
import os
import sys
import gc
import tracemalloc
from typing import Dict, List, Tuple
import numpy as np

def format_size(bytes_size):
    """Format bytes to human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.2f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.2f} PB"

def get_memory_info() -> Dict:
    """Get current memory usage information"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    system_mem = psutil.virtual_memory()
    
    return {
        'rss': mem_info.rss,  # Resident Set Size (actual RAM used)
        'vms': mem_info.vms,  # Virtual Memory Size
        'percent': process.memory_percent(),
        'system_total': system_mem.total,
        'system_available': system_mem.available,
        'system_percent': system_mem.percent,
        'system_used': system_mem.used
    }

def print_memory_status(label: str = ""):
    """Print current memory status"""
    mem = get_memory_info()
    print(f"\n{'='*60}")
    if label:
        print(f"Memory Status: {label}")
    else:
        print("Memory Status")
    print(f"{'='*60}")
    print(f"Process RSS (actual RAM): {format_size(mem['rss'])} ({mem['percent']:.2f}%)")
    print(f"Process VMS (virtual): {format_size(mem['vms'])}")
    print(f"System Total RAM: {format_size(mem['system_total'])}")
    print(f"System Available: {format_size(mem['system_available'])} ({100 - mem['system_percent']:.2f}% free)")
    print(f"System Used: {format_size(mem['system_used'])} ({mem['system_percent']:.2f}%)")
    print(f"{'='*60}\n")

def estimate_tile_memory(tile_size: Tuple[int, int, int], n_tiles: int) -> Dict:
    """Estimate memory needed for tiles"""
    z, y, x = tile_size
    bytes_per_tile = z * y * x * 4  # float32 = 4 bytes
    total_bytes = bytes_per_tile * n_tiles
    
    return {
        'bytes_per_tile': bytes_per_tile,
        'total_bytes': total_bytes,
        'n_tiles': n_tiles,
        'tile_size': tile_size
    }

def check_memory_before_processing(
    tile_size: Tuple[int, int, int],
    n_images: int,
    estimated_tiles_per_image: int = 1000,
    safety_margin: float = 0.5
) -> Tuple[bool, str]:
    """
    Check if we have enough memory before starting processing.
    
    Returns:
        (can_proceed, message)
    """
    mem = get_memory_info()
    available_mb = mem['system_available'] / (1024 * 1024)
    
    # Estimate memory needs
    total_estimated_tiles = n_images * estimated_tiles_per_image
    tile_est = estimate_tile_memory(tile_size, total_estimated_tiles)
    
    # Memory for:
    # 1. One 2GB image in memory-mapped form (minimal, but some overhead)
    # 2. Tiles being processed (batch)
    # 3. Coordinates
    # 4. System overhead
    
    image_overhead_mb = 100  # Memory-mapped overhead for 2GB image
    batch_tiles_mb = (tile_est['bytes_per_tile'] * 1000) / (1024 * 1024)  # 1000 tiles in batch
    coords_mb = total_estimated_tiles * 0.01  # Rough estimate for coordinates
    system_overhead_mb = 5000  # 5GB for system
    
    total_needed_mb = image_overhead_mb + batch_tiles_mb + coords_mb + system_overhead_mb
    total_needed_mb *= (1 + safety_margin)  # Add safety margin
    
    can_proceed = available_mb >= total_needed_mb
    
    message = f"""
Memory Check:
  Available RAM: {format_size(mem['system_available'])}
  Estimated needed: {format_size(total_needed_mb * 1024 * 1024)}
  Tile size: {tile_size}
  Estimated tiles: {total_estimated_tiles}
  Per tile: {format_size(tile_est['bytes_per_tile'])}
  Total tile data: {format_size(tile_est['total_bytes'])}
  
  Breakdown:
    - Image overhead: {format_size(image_overhead_mb * 1024 * 1024)}
    - Batch tiles (1000): {format_size(batch_tiles_mb * 1024 * 1024)}
    - Coordinates: {format_size(coords_mb * 1024 * 1024)}
    - System overhead: {format_size(system_overhead_mb * 1024 * 1024)}
    - Safety margin ({safety_margin*100}%): {format_size((total_needed_mb * safety_margin) * 1024 * 1024)}
  
  Status: {'✓ OK' if can_proceed else '✗ INSUFFICIENT MEMORY'}
"""
    
    return can_proceed, message

def monitor_memory_usage(func, *args, **kwargs):
    """Decorator to monitor memory usage of a function"""
    def wrapper(*args, **kwargs):
        mem_before = get_memory_info()
        print_memory_status("Before function call")
        
        tracemalloc.start()
        try:
            result = func(*args, **kwargs)
            mem_after = get_memory_info()
            snapshot = tracemalloc.take_snapshot()
            
            print_memory_status("After function call")
            print(f"Memory increase: {format_size(mem_after['rss'] - mem_before['rss'])}")
            
            # Show top allocations
            top_stats = snapshot.statistics('lineno')
            print("\nTop 5 memory allocations:")
            for index, stat in enumerate(top_stats[:5], 1):
                print(f"  {index}. {format_size(stat.size)} - {stat.traceback[0].filename}:{stat.traceback[0].lineno}")
            
            return result
        finally:
            tracemalloc.stop()
            gc.collect()
    
    return wrapper

def force_garbage_collection():
    """Force aggressive garbage collection"""
    collected = gc.collect()
    return collected

def get_largest_objects(n: int = 10) -> List[Tuple[str, int]]:
    """Get the largest objects in memory"""
    import sys
    objects = []
    for obj in gc.get_objects():
        try:
            size = sys.getsizeof(obj)
            obj_type = type(obj).__name__
            objects.append((f"{obj_type}", size))
        except:
            pass
    
    # Sort by size and return top N
    objects.sort(key=lambda x: x[1], reverse=True)
    return objects[:n]

def diagnose_memory_issue():
    """Diagnose current memory issues"""
    print("\n" + "="*60)
    print("MEMORY DIAGNOSIS")
    print("="*60)
    
    mem = get_memory_info()
    print_memory_status("Current")
    
    # Check if we're close to limit
    if mem['system_percent'] > 90:
        print("⚠️  WARNING: System memory usage > 90%")
    if mem['percent'] > 50:
        print(f"⚠️  WARNING: Process using {mem['percent']:.2f}% of system RAM")
    
    # Force GC
    print("\nForcing garbage collection...")
    collected = force_garbage_collection()
    print(f"Collected {collected} objects")
    
    mem_after_gc = get_memory_info()
    freed = mem['rss'] - mem_after_gc['rss']
    if freed > 0:
        print(f"Freed {format_size(freed)} of memory")
    
    # Show largest objects
    print("\nLargest objects in memory:")
    largest = get_largest_objects(10)
    for i, (obj_type, size) in enumerate(largest, 1):
        print(f"  {i}. {obj_type}: {format_size(size)}")
    
    print("\n" + "="*60)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Memory monitoring and diagnosis')
    parser.add_argument('--check', action='store_true', help='Check memory before processing')
    parser.add_argument('--tile-size', type=int, nargs=3, default=[8, 64, 64], help='Tile size (z y x)')
    parser.add_argument('--n-images', type=int, default=12, help='Number of images')
    parser.add_argument('--tiles-per-image', type=int, default=100, help='Estimated tiles per image')
    parser.add_argument('--diagnose', action='store_true', help='Diagnose current memory issues')
    
    args = parser.parse_args()
    
    if args.diagnose:
        diagnose_memory_issue()
    elif args.check:
        can_proceed, message = check_memory_before_processing(
            tuple(args.tile_size),
            args.n_images,
            args.tiles_per_image
        )
        print(message)
        sys.exit(0 if can_proceed else 1)
    else:
        print_memory_status("Current")
